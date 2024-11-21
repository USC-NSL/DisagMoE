#include <iostream>
#include <fstream>
#include <string>
#include <map>

#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

#include "constants.h"
#include "logging.h"

static std::map<int, std::string> device_id_2_ip;

// this function must be called before init engine
// NOTE(hogura|20241120): local_device_id is mapped to 0.0.0.0, as in engine.py:set_hosts
static void set_hosts_internal(int process_id, const std::map<int, std::string>& device_id_2_ip_) {
    // !NOTE(hogura|20241120): this storage could not work since this variable is not shared across shared libraries
    device_id_2_ip = device_id_2_ip_;
    DMOE_LOG(ERROR) << "set_hosts_internal " << device_id_2_ip.size() << LEND;

    // we have to write the config into files
    mkdir(TEMP_DIR, S_IRWXU | S_IRWXG | S_IRWXO);
    std::string filename = std::string(TEMP_DIR) + "hostfile_" + std::to_string(process_id);
    std::ofstream ofs(filename, std::ios::trunc | std::ios::out);
    for (auto &pr: device_id_2_ip) {
        ofs << pr.first << " " << pr.second << std::endl;
    }
    ofs.close();
}

static std::string get_ip_of_device(int device_id) {
    if (device_id_2_ip.empty()) {
        auto pid = getpid();
        std::string filename = std::string(TEMP_DIR) + "hostfile_" + std::to_string(pid);
        std::ifstream ifs(filename);
        if (!ifs.is_open()) {
            DMOE_LOG(ERROR) << "file " << filename << " not found" << LEND;
        }
        int id;
        std::string ip;
        while (ifs >> id >> ip) {
            DMOE_LOG(WARNING) << "reading device_id " << id << " ip: " << ip << LEND;
            device_id_2_ip[id] = ip;
        }
        ifs.close();
    }
    if (device_id_2_ip.find(device_id) == device_id_2_ip.end()) {
        DMOE_LOG(ERROR) << "device_id " << device_id << " not found in device_id_2_ip, with size: " << device_id_2_ip.size() << LEND;
    } else {
        DMOE_LOG(WARNING) << "device_id " << device_id << " ip: " << device_id_2_ip.at(device_id) << LEND;
    }
    return device_id_2_ip.at(device_id);
}

inline std::string get_zmq_addr(int device_id, bool is_gpu = true, int manual_port = -1) {
    int port = device_id +
        (manual_port == -1 \
            ? (is_gpu ? ZMQ_PORT_BASE : ZMQ_CPU_PORT_BASE)
            : manual_port);
    fprintf(stderr, "zmq device_id: %d, port: %d, manual port %d\n", device_id, port, manual_port);
    std::string ip = get_ip_of_device(device_id);
    return "tcp://" + ip + ":" + std::to_string(port);
}