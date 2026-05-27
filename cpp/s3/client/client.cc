
#include <aws/s3-crt/model/GetObjectRequest.h>

#include <cstring>
#include <algorithm>
#include <string>
#include <utility>
#include <optional>
#include <chrono>
#include <fstream>
#include <mutex>
#include <thread>
#include <atomic>

#include "common/backend_api/object_storage/object_storage.h"
#include "s3/client/client.h"

#include "common/exception/exception.h"

#include "utils/logging/logging.h"
#include "utils/env/env.h"
#include "utils/fd/fd.h"

namespace {

struct TraceLog {
    std::mutex mu;
    std::ofstream file;
    std::chrono::steady_clock::time_point epoch;
    std::atomic<bool> conn_monitor_stop{false};
    std::thread conn_thread;

    TraceLog() : epoch(std::chrono::steady_clock::now()) {
        const char* path = std::getenv("RUNAI_STREAMER_TRACE_FILE");
        if (path) {
            file.open(path, std::ios::out | std::ios::trunc);
            if (file.is_open()) {
                file << "event,timestamp_ms,object_key,offset,length,latency_ms,error\n";
                conn_thread = std::thread(&TraceLog::monitor_connections, this);
            }
        }
    }

    ~TraceLog() {
        conn_monitor_stop = true;
        if (conn_thread.joinable()) conn_thread.join();
    }

    void monitor_connections() {
        while (!conn_monitor_stop) {
            int count = 0;
            std::ifstream tcp("/proc/net/tcp");
            std::ifstream tcp6("/proc/net/tcp6");
            std::string line;
            // Count established (state 01) connections to port 443 (01BB hex)
            auto count_file = [&](std::ifstream& f) {
                while (std::getline(f, line)) {
                    // remote_address field is at fixed position, port is after ':'
                    // Format: " sl  local_address rem_address   st ..."
                    if (line.find(":01BB ") != std::string::npos ||
                        line.find(":01BB ") != std::string::npos) {
                        // Check state field (column 4, should be "01" for ESTABLISHED)
                        auto st_pos = line.find(" 01 ", 20);
                        if (st_pos != std::string::npos) count++;
                    }
                }
            };
            count_file(tcp);
            count_file(tcp6);

            auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - epoch).count();
            {
                std::lock_guard<std::mutex> lock(mu);
                if (file.is_open()) {
                    file << "CONN," << ts << "," << count << "\n";
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    void log_get(const std::string& key, size_t offset, size_t length,
                 std::chrono::steady_clock::time_point start, bool error) {
        if (!file.is_open()) return;
        auto now = std::chrono::steady_clock::now();
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(now - epoch).count();
        auto latency = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        std::lock_guard<std::mutex> lock(mu);
        file << "GET," << ts << "," << key << "," << offset << "," << length
             << "," << latency << "," << (error ? 1 : 0) << "\n";
    }

    static TraceLog& instance() {
        static TraceLog tl;
        return tl;
    }
};

} // anonymous namespace

namespace runai::llm::streamer::impl::s3
{

std::optional<Aws::String> convert(const char * input)
{
    std::optional<Aws::String> result = std::nullopt;
    if (input)
    {
        result = Aws::String(input);
    }
    return result;
}

S3ClientBase::S3ClientBase(const common::backend_api::ObjectClientConfig_t & config) :
    _endpoint(convert(config.endpoint_url)),
    _chunk_bytesize(config.default_storage_chunk_size)
{
    auto ptr = config.initial_params;
    if (ptr)
    {
        for (size_t i = 0; i < config.num_initial_params; ++i, ++ptr)
        {
            const char* key = ptr->key;
            const char* value = ptr->value;
            if (strcmp(key, common::s3::Credentials::ACCESS_KEY_ID_KEY) == 0)
            {
                _key = convert(value);
            }
            else if (strcmp(key, common::s3::Credentials::SECRET_ACCESS_KEY_KEY) == 0)
            {
                _secret = convert(value);
            }
            else if (strcmp(key, common::s3::Credentials::SESSION_TOKEN_KEY) == 0)
            {
                _token = convert(value);
            }
            else if (strcmp(key, common::s3::Credentials::REGION_KEY) == 0)
            {
                _region = convert(value);
            }
            else
            {
                LOG(WARNING) << "Unknown initial parameter: " << key;
            }
        }
    }
}

bool S3ClientBase::verify_credentials_member(const std::optional<Aws::String>& member, const std::optional<Aws::String>& value, const char * name) const
{
    if (member.has_value())
    {
        if (!value.has_value())
        {
            LOG(DEBUG) << "credentials member " << name << " is set, but provided member is nullptr";
            return false;
        }
        if (member.value() != value.value())
        {
            LOG(DEBUG) << "credentials member " << name << " doesn't match the provided value";
            return false;
        }
    }
    else if (value.has_value()) // must be nullptr and not empty string
    {
        LOG(DEBUG) << "credentials member " << name << " is not set, but provided member is not nullptr";
        return false;
    }
    LOG(DEBUG) << "credentials member " << name << " verified";
    return true;
}

bool S3ClientBase::verify_credentials(const common::backend_api::ObjectClientConfig_t & config) const
{
    S3ClientBase other(config);
    return (verify_credentials_member(_key, other._key, "access key") &&
            verify_credentials_member(_secret, other._secret, "secret") &&
            verify_credentials_member(_token, other._token, "session token") &&
            verify_credentials_member(_region, other._region, "region") &&
            verify_credentials_member(_endpoint, other._endpoint, "endpoint"));
}

S3Client::S3Client(const common::backend_api::ObjectClientConfig_t & config) :
    S3ClientBase(config),
    _stop(false),
    _responder(nullptr)
{
    if (_endpoint.has_value()) // endpoint passed as parameter by user application (in credentials)
    {
        _client_config.config.endpointOverride = _endpoint.value();
    }

    if (utils::try_getenv("RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING", _client_config.config.useVirtualAddressing))
    {
        LOG(DEBUG) << "Setting s3 configuration useVirtualAddressing to " << _client_config.config.useVirtualAddressing;
    }

    if (_region.has_value())
    {
        LOG(DEBUG) << "Setting s3 region to " << _region.value();
        _client_config.config.region = _region.value();
    }

    if (utils::try_getenv("AWS_CA_BUNDLE", _client_config.config.caFile))
    {
        LOG(DEBUG) << "Setting s3 configuration ca certificate file to " << _client_config.config.caFile;

        // verify file exists
        if (!utils::Fd::exists(_client_config.config.caFile))
        {
            LOG(ERROR) << "CA cert file not found: " << _client_config.config.caFile;
            throw common::Exception(common::ResponseCode::CaFileNotFound);
        }
    }

    if (_client_credentials == nullptr)
    {
        _client = std::make_unique<Aws::S3Crt::S3CrtClient>(_client_config.config);
        LOG(DEBUG) << "Using default authentication";
    }
    else
    {
        LOG(DEBUG) << "Creating S3 client with given credentials";
        _client = std::make_unique<Aws::S3Crt::S3CrtClient>(*_client_credentials, _client_config.config);
    }
}

// returns response object that contains the index of the range in ranges vector  which was passed in the request (0... number of ranges - 1)
common::backend_api::Response S3Client::async_read_response()
{
    if (_responder == nullptr)
    {
        LOG(WARNING) << "Requesting response with uninitialized responder";
        return common::ResponseCode::FinishedError;
    }

    return _responder->pop();
}


common::backend_api::ResponseCode_t S3Client::async_read(const char* path,
                                                         common::backend_api::ObjectRange_t range,
                                                         char* destination_buffer,
                                                         common::backend_api::ObjectRequestId_t request_id)
{
    if (_responder == nullptr)
    {
        _responder = std::make_shared<Responder>(1);
    }
    else
    {
        _responder->increment(1);
    }

    const auto uri = common::s3::StorageUri(path);

    Aws::String bucket_name(uri.bucket);
    Aws::String path_name(uri.path);

    char * buffer_ = destination_buffer;
    // split range into chunks
    size_t size = std::max(1UL, range.length/_chunk_bytesize);
    LOG(SPAM) <<"Number of chunks is " << size;

    // each range is divided into chunks (size is the number of chunks)
    // when all the chunks have been read successfuly the response for that range is pushed to the responder

    auto counter = std::make_shared< std::atomic<unsigned> >(size);
    // success flag for the current range is passed to the client
    auto is_success = std::make_shared< std::atomic<bool> >(true);

    size_t total_ = range.length;
    size_t offset_ = range.offset;
    for (unsigned i = 0; i < size && !_stop; ++i)
    {
        size_t bytesize_ = (i == size - 1 ? total_ : _chunk_bytesize);

        // send async request
        auto request = std::make_shared<Aws::S3Crt::Model::GetObjectRequest>();

        request->SetBucket(bucket_name);
        request->SetKey(path_name);
        std::string range_str = "bytes=" + std::to_string(offset_) + "-" + std::to_string(offset_ + bytesize_ - 1);
        request->SetRange(range_str.c_str());

        request->SetResponseStreamFactory(
            [buffer_, bytesize_]()
            {
                std::unique_ptr<Aws::StringStream>
                        stream(Aws::New<Aws::StringStream>("RunaiBuffer"));

                stream->rdbuf()->pubsetbuf(buffer_, bytesize_);

                return stream.release();
            });

        auto trace_start = std::chrono::steady_clock::now();
        auto trace_key = std::string(path_name.c_str());
        auto trace_offset = offset_;
        auto trace_length = bytesize_;

        _client->GetObjectAsync(*request, [request, responder = _responder, request_id, counter, is_success,
                                           trace_start, trace_key, trace_offset, trace_length](
                                                                        const Aws::S3Crt::S3CrtClient*, const Aws::S3Crt::Model::GetObjectRequest&,
                                                                        const Aws::S3Crt::Model::GetObjectOutcome& outcome,
                                                                        const std::shared_ptr<const Aws::Client::AsyncCallerContext>&) {
            bool err = !outcome.IsSuccess();
            TraceLog::instance().log_get(trace_key, trace_offset, trace_length, trace_start, err);

            if (outcome.IsSuccess())
            {
                const auto running = counter->fetch_sub(1);
                LOG(SPAM) << "Async read request " << request_id << " succeeded - " << running << " running";
                // send success response only if all the requests have succeeded
                // note that unsuccessful attempts do not update the counter
                if (running == 1)
                {
                    common::backend_api::Response r(request_id, common::ResponseCode::Success);
                    responder->push(std::move(r));
                }
            }
            else
            {
                // Note: currently a failure to read any sub range fails the entire read request
                //       a retry mechanism should be added for failed reads
                bool previous = is_success->exchange(false);
                // send error response only once
                if (previous)
                {
                    const auto & err = outcome.GetError();
                    LOG(ERROR) << "Failed to download s3 object of request " << request_id << " " << err.GetExceptionName() << ": " << err.GetMessage();
                    common::backend_api::Response r(request_id, common::ResponseCode::FileAccessError);
                    responder->push(std::move(r));
                }
            }
        });

        total_ -= bytesize_;
        offset_ += bytesize_;
        buffer_ += bytesize_;
    }

    return _stop ? common::ResponseCode::FinishedError : common::ResponseCode::Success;
}

void S3Client::stop()
{
    _stop = true;
    if (_responder != nullptr)
    {
        _responder->stop();
    }
}

}; // namespace runai::llm::streamer::impl::s3
