#pragma once

#ifndef POOL_H_
#define POOL_H_

#include "muhelper.h"
#include "datatypes.hpp"
#include "metadata.hpp"
#include "batch.hpp"
#include "layer.h"

class UnifiedPool: public MuPool {

private:

    std::vector<UnifiedLayer> layers;

};

#endif