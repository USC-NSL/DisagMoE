#include "embedding.h"

Sampler::Sampler(int device_id, std::vector<Channel_t> channels):
    MuHelper({}, device_id, channels) {

    }

void Sampler::run() {
    while (!this->end_flag) {
        
    }
}