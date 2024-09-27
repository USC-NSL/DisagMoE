#include "comm.h"

NcclChannel::NcclChannel(int party_local, int party_other, ncclUniqueId comm_id): Channel::Channel(party_local, party_other), comm_id(comm_id) 
    {
        CUDACHECK(cudaStreamCreate(&this->stream));
        puts("Created a cuda stream");
        printf("%d %d\n", this->local, this->other);
    }

NcclChannel::~NcclChannel() {
    CUDACHECK(cudaStreamDestroy(this->stream));
}

void NcclChannel::instantiate() {
    printf("calling nccl init (%d, %d) %d\n", this->local, this->other, this->m_rank());
    CUDACHECK(cudaSetDevice(this->local));
    NCCLCHECK(ncclCommInitRank(
        &this->comm,
        /*nranks=*/ 2,
        this->comm_id,
        /*rank=*/ this->m_rank()
    ));
    puts("called nccl init");
}

void NcclChannel::send(void* data, const Metadata& metadata) {
    // TODO(hogura|20240926): check if ncclGroupStart may influence the performance
    NCCLCHECK(ncclSend(
        data, 
        /*count=*/ metadata.get_num_element(),
        /*datatype=*/ metadata.get_nccl_datatype(),
        this->other,
        this->comm,
        this->stream
    ));
}

void NcclChannel::recv(void* data, const Metadata& metadata) {
    NCCLCHECK(ncclRecv(
        data,
        /*count=*/ metadata.get_num_element(),
        /*datatype=*/ metadata.get_nccl_datatype(),
        this->other,
        this->comm,
        this->stream
    ));
}


Channel_t create_channel(int party_local, int party_other, void *nccl_id_raw) {
    ncclUniqueId& id = *((ncclUniqueId*)(nccl_id_raw));
    auto channel = std::make_shared<NcclChannel>(
        party_local, party_other, id
    );
    // TODO(hogura|20240927): recycle the ncclUniqueId (raw).
    return channel;
}

void* get_nccl_unique_id() {
    void* _data = std::malloc(sizeof(ncclUniqueId));
    ncclGetUniqueId((ncclUniqueId*)_data);
    return _data;
}

void instantiate_channels(std::vector<Channel_t> channels) {
    std::vector<std::thread> threads;
    puts("creating channels");
    for (auto c: channels) {
        c->_debug_print();
        threads.push_back(std::thread([=](auto channel) {channel->instantiate();}, c));
    }
    for (auto &t: threads) {
        t.join();
    }
    puts("threads inited");
}