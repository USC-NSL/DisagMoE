#include <iostream>
#include <fstream>

int main() {
    std::ifstream file("/tmp/disagmoe/hostfile_387214");
    int id;
    std::string ip;
    while (file >> id >> ip) {
        std::cout << id << " " << ip << std::endl;
    }
    return 0;
}