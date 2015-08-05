require 'cudnn'
require 'cunn'
local SpatialConvolution = cudnn.SpatialConvolution--lib[1]
local SpatialMaxPooling = cudnn.SpatialMaxPooling--lib[2]

-- #classes
nC = 333

-- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
-- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
local features = nn.Sequential()
features:add(SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
features:add(cudnn.ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
features:add(SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
features:add(cudnn.ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
features:add(SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
features:add(cudnn.ReLU(true))
features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
features:add(cudnn.ReLU(true))
features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
features:add(cudnn.ReLU(true))
features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

local classifier = nn.Sequential()
classifier:add(nn.View(256*6*6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*6*6, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.Threshold(0, 1e-6))
classifier:add(nn.Linear(4096, nC))
classifier:add(nn.LogSoftMax())

local model = nn.Sequential()

function fillBias(m)
  for i=1, #m.modules do
    if m:get(i).bias then
      m:get(i).bias:fill(0.1)
    end
  end
end

fillBias(features)
fillBias(classifier)
model:add(features):add(classifier)

-- learning rate
local nEpo = 60
local lrs = {
  { 1,   18, 1e-2, 5e-4,},
  {19,   29, 5e-3, 5e-4 },
  {30,   43, 1e-3, 0},
  {44,   52, 5e-4, 0},
  {53, nEpo, 1e-4, 0},
}

return {model = model,
        nEpo = nEpo,
        lrs = lrs}
