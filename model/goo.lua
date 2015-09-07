#!/usr/bin/env th
-- GoogLetNet Model.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 2015-08
--   modify  -  Feng Zhou (zhfe99@gmail.com), 2015-09

require 'cudnn'
require 'nn'
require 'nngraph'
local lib = require('lua_lib')
local th = require('lua_th')
local goo = {}
local modPath0 = paths.concat(paths.home, 'save/imgnet/torch/model/imgnet_v2_goo_4gpu.t7_33.t7')


----------------------------------------------------------------------
-- Create the basic GoogLeNet model.
--
-- Input
--   nC     -  #classes
--   bn     -  type of BN
--   ini    -  init method
--
-- Output
--   model  -  model
function goo.newc(nC, bn, ini)
  local part1 = nn.Sequential()
  -- conv1
  part1:add(cudnn.SpatialConvolution(3,64,7,7,2,2,3,3))
  part1:add(cudnn.ReLU(true))
  part1:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())

  -- conv2 a
  if bn then
    part1:add(nn.SpatialBatchNormalization(64, nil, nil, false))
  end
  part1:add(cudnn.SpatialConvolution(64,64,1,1))
  part1:add(cudnn.ReLU(true))

  -- conv2 b
  if bn then
    part1:add(nn.SpatialBatchNormalization(64, nil, nil, false))
  end
  part1:add(cudnn.SpatialConvolution(64,192,3,3,1,1,1,1))
  part1:add(cudnn.ReLU(true))
  part1:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())

  if bn then
    part1:add(nn.SpatialBatchNormalization(192, nil, nil, false))
  end
  part1:add(inceptionc(192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}}, bn)) -- 3(a)

  if bn then
    part1:add(nn.SpatialBatchNormalization(256, nil, nil, false))
  end
  part1:add(inceptionc(256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}}, bn)) -- 3(b)

  if bn then
    part1:add(nn.SpatialBatchNormalization(320, nil, nil, false))
  end
  part1:add(inceptionc(320, {{  0},{128,160},{ 64, 96},{'max',  0}}, bn)) -- 3(c)
  part1:add(cudnn.SpatialConvolution(576,576,2,2,2,2))
  if bn then
    part1:add(nn.SpatialBatchNormalization(576, nil, nil, false))
  end

  -- part2
  local part2 = nn.Sequential()
  part2:add(inceptionc(576, {{224},{ 64, 96},{ 96,128},{'avg',128}}, bn)) -- 4(a)
  if bn then
    part2:add(nn.SpatialBatchNormalization(576, nil, nil, false))
  end
  part2:add(inceptionc(576, {{192},{ 96,128},{ 96,128},{'avg',128}}, bn)) -- 4(b)
  if bn then
    part2:add(nn.SpatialBatchNormalization(576, nil, nil, false))
  end
  part2:add(inceptionc(576, {{160},{128,160},{128,160},{'avg', 96}}, bn)) -- 4(c)
  if bn then
    part2:add(nn.SpatialBatchNormalization(576, nil, nil, false))
  end

  -- part3
  local part3 = nn.Sequential()
  part3:add(inceptionc(576, {{ 96},{128,192},{160,192},{'avg', 96}}, bn)) -- 4(d)
  if bn then
    part3:add(nn.SpatialBatchNormalization(576, nil, nil, false))
  end
  part3:add(inceptionc(576, {{  0},{128,192},{192,256},{'max', 0}}, bn)) -- 4(e)
  if bn then
    part3:add(nn.SpatialBatchNormalization(1024, nil, nil, false))
  end
  part3:add(cudnn.SpatialConvolution(1024,1024,2,2,2,2))
  part3:add(inceptionc(1024, {{352},{192,320},{160,224},{'avg',128}}, bn)) -- 5(a)
  if bn then
    part3:add(nn.SpatialBatchNormalization(1024, nil, nil, false))
  end
  part3:add(inceptionc(1024, {{352},{192,320},{192,224},{'max',128}}, bn)) -- 5(b)
  -- if bn then
  --   part3:add(nn.SpatialBatchNormalization(1024, nil, nil, false))
  -- end

  -- classifier 1
  local clfy1 = nn.Sequential()
  clfy1:add(cudnn.SpatialAveragePooling(7,7,1,1))
  clfy1:add(nn.View(1024):setNumInputDims(3))
  clfy1:add(nn.Linear(1024, nC))
  clfy1:add(nn.LogSoftMax())

  -- classifier 2
  local clfy2 = nn.Sequential()
  clfy2:add(cudnn.SpatialAveragePooling(5,5,3,3):ceil())
  clfy2:add(cudnn.SpatialConvolution(576,128,1,1,1,1))
  clfy2:add(cudnn.ReLU(true))
  clfy2:add(nn.View(128*4*4):setNumInputDims(3))
  clfy2:add(nn.Linear(128*4*4,768))
  clfy2:add(nn.ReLU(true))
  clfy2:add(nn.Linear(768, nC))
  clfy2:add(nn.LogSoftMax())

  -- classifier 3
  local clfy3 = clfy2:clone()

  -- combine
  local input = nn.Identity()()
  local output1 = part1(input)
  local branch1 = clfy2(output1)

  local output2 = part2(output1)
  local branch2 = clfy3(output2)

  local mainBranch = clfy1(part3(output2))
  local model = nn.gModule({input}, {mainBranch, branch1, branch2})

  -- init
  th.iniMod(model, ini)

  return model, {{}}
end

----------------------------------------------------------------------
-- Create the inception component.
--
-- Input
--   k0       -  input dimension
--   config   -  configuration
--   bn       -  type of BN
--
-- Output
--   concat   -  inception module
local function inceptionb(k0, config, bn)
  local concat = nn.Concat(2)
  if config[1][1] ~= 0 then
    local conv1 = nn.Sequential()
    conv1:add(cudnn.SpatialConvolution(k0, config[1][1], 1, 1, 1, 1))
    th.addSBN(conv1, config[1][1], bn)
    conv1:add(cudnn.ReLU(true))
    concat:add(conv1)
  end

  local conv3 = nn.Sequential()
  conv3:add(cudnn.SpatialConvolution(k0, config[2][1],1,1,1,1))
  th.addSBN(conv3, config[2][1], bn)
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1))
  th.addSBN(conv3, config[2][2], bn)
  conv3:add(cudnn.ReLU(true))
  concat:add(conv3)

  local conv3xx = nn.Sequential()
  conv3xx:add(cudnn.SpatialConvolution(k0, config[3][1],1,1,1,1))
  th.addSBN(conv3xx, config[3][1], bn)
  conv3xx:add(cudnn.ReLU(true))
  conv3xx:add(cudnn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1))
  th.addSBN(conv3xx, config[3][2], bn)
  conv3xx:add(cudnn.ReLU(true))
  conv3xx:add(cudnn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1))
  th.addSBN(conv3xx, config[3][2], bn)
  conv3xx:add(cudnn.ReLU(true))
  concat:add(conv3xx)

  local pool = nn.Sequential()
  pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting cudnn R2 into fbcode
  if config[4][1] == 'max' then
    pool:add(cudnn.SpatialMaxPooling(3,3,1,1):ceil())
  elseif config[4][1] == 'avg' then
    pool:add(cudnn.SpatialAveragePooling(3,3,1,1):ceil())
  else
    error('Unknown pooling')
  end
  if config[4][2] ~= 0 then
    pool:add(cudnn.SpatialConvolution(k0,config[4][2],1,1,1,1))
    th.addSBN(pool, config[4][2], bn)
    pool:add(cudnn.ReLU(true))
  end
  concat:add(pool)

  return concat
end

----------------------------------------------------------------------
-- Create the basic GoogLeNet model.
--
-- Input
--   nC     -  #classes
--   bn     -  type of BN
--   ini    -  init method
--
-- Output
--   model  -  model
function goo.newb(nC, bn, ini)
  local features = nn.Sequential()
  features:add(cudnn.SpatialConvolution(3,64,7,7,2,2,3,3))
  th.addSBN(features, 64, bn)
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
  features:add(cudnn.SpatialConvolution(64,64,1,1))
  th.addSBN(features, 64, bn)
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialConvolution(64,192,3,3,1,1,1,1))
  th.addSBN(features, 192, bn)
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
  features:add(inceptionb(192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}}, bn)) -- 3(a)
  features:add(inceptionb(256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}}, bn)) -- 3(b)
  features:add(inceptionb(320, {{  0},{128,160},{ 64, 96},{'max',  0}}, bn)) -- 3(c)
  features:add(cudnn.SpatialConvolution(576,576,2,2,2,2))

  features:add(inceptionb(576, {{224},{ 64, 96},{ 96,128},{'avg',128}}, bn)) -- 4(a)
  features:add(inceptionb(576, {{192},{ 96,128},{ 96,128},{'avg',128}}, bn)) -- 4(b)
  features:add(inceptionb(576, {{160},{128,160},{128,160},{'avg', 96}}, bn)) -- 4(c)
  features:add(inceptionb(576, {{ 96},{128,192},{160,192},{'avg', 96}}, bn)) -- 4(d)

  local main_branch = nn.Sequential()
  main_branch:add(inceptionb(576, {{  0},{128,192},{192,256},{'max', 0}}, bn)) -- 4(e)
  main_branch:add(cudnn.SpatialConvolution(1024,1024,2,2,2,2))

  main_branch:add(inceptionb(1024, {{352},{192,320},{160,224},{'avg',128}}, bn)) -- 5(a)
  main_branch:add(inceptionb(1024, {{352},{192,320},{192,224},{'max',128}}, bn)) -- 5(b)
  main_branch:add(cudnn.SpatialAveragePooling(7,7,1,1))
  main_branch:add(nn.View(1024):setNumInputDims(3))
  main_branch:add(nn.Linear(1024, nC))
  main_branch:add(nn.LogSoftMax())

  local model = nn.Sequential():add(features):add(main_branch)

  -- init
  th.iniMod(model, ini)

  return model, {}
end

----------------------------------------------------------------------
-- Create the inception component.
--
-- Input
--   k0       -  input dimension
--   config   -  configuration
--   bn       -  type of BN
--
-- Output
--   concat   -  inception module
local function inception(k0, config, bn)
  local concat = nn.Concat(2)
  if config[1][1] ~= 0 then
    local conv1 = nn.Sequential()
    -- to del
    th.addSBN(conv1, 96, bn)

    conv1:add(cudnn.SpatialConvolution(k0, config[1][1], 1, 1, 1, 1))
    th.addSBN(conv1, config[1][1], bn)
    conv1:add(cudnn.ReLU(true))
    concat:add(conv1)
  end

  local conv3 = nn.Sequential()
  conv3:add(cudnn.SpatialConvolution(k0, config[2][1],1,1,1,1))
  th.addSBN(conv3, config[2][1], bn)
  conv3:add(cudnn.ReLU(true))
  conv3:add(cudnn.SpatialConvolution(config[2][1], config[2][2],3,3,1,1,1,1))
  th.addSBN(conv3, config[2][2], bn)
  conv3:add(cudnn.ReLU(true))
  concat:add(conv3)

  local conv3xx = nn.Sequential()
  conv3xx:add(cudnn.SpatialConvolution(k0, config[3][1],1,1,1,1))
  th.addSBN(conv3xx, config[3][1], bn)
  conv3xx:add(cudnn.ReLU(true))
  conv3xx:add(cudnn.SpatialConvolution(config[3][1], config[3][2],3,3,1,1,1,1))
  th.addSBN(conv3xx, config[3][2], bn)
  conv3xx:add(cudnn.ReLU(true))
  conv3xx:add(cudnn.SpatialConvolution(config[3][2], config[3][2],3,3,1,1,1,1))
  th.addSBN(conv3xx, config[3][2], bn)
  conv3xx:add(cudnn.ReLU(true))
  concat:add(conv3xx)

  local pool = nn.Sequential()
  pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting cudnn R2 into fbcode
  if config[4][1] == 'max' then
    pool:add(cudnn.SpatialMaxPooling(3,3,1,1):ceil())
  elseif config[4][1] == 'avg' then
    pool:add(cudnn.SpatialAveragePooling(3,3,1,1):ceil())
  else
    error('Unknown pooling')
  end
  if config[4][2] ~= 0 then
    pool:add(cudnn.SpatialConvolution(k0,config[4][2],1,1,1,1))
    th.addSBN(pool, config[4][2], bn)
    pool:add(cudnn.ReLU(true))
  end
  concat:add(pool)

  return concat
end

----------------------------------------------------------------------
-- Create the basic GoogLeNet model.
--
-- Input
--   nC     -  #classes
--   bn     -  type of BN
--   ini    -  init method
--
-- Output
--   model  -  model
function goo.new(nC, bn, ini)
  local features = nn.Sequential()
  features:add(cudnn.SpatialConvolution(3,64,7,7,2,2,3,3))
  th.addSBN(features, 64, bn)
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
  features:add(cudnn.SpatialConvolution(64,64,1,1))
  th.addSBN(features, 64, bn)
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialConvolution(64,192,3,3,1,1,1,1))
  th.addSBN(features, 192, bn)
  features:add(cudnn.ReLU(true))
  features:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
  features:add(inception(192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}}, bn)) -- 3(a)
  features:add(inception(256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}}, bn)) -- 3(b)
  features:add(inception(320, {{  0},{128,160},{ 64, 96},{'max',  0}}, bn)) -- 3(c)
  features:add(cudnn.SpatialConvolution(576,576,2,2,2,2))

  -- to del
  th.addSBN(features, 576, bn)

  features:add(inception(576, {{224},{ 64, 96},{ 96,128},{'avg',128}}, bn)) -- 4(a)
  features:add(inception(576, {{192},{ 96,128},{ 96,128},{'avg',128}}, bn)) -- 4(b)
  features:add(inception(576, {{160},{128,160},{128,160},{'avg', 96}}, bn)) -- 4(c)
  features:add(inception(576, {{ 96},{128,192},{160,192},{'avg', 96}}, bn)) -- 4(d)

  local main_branch = nn.Sequential()
  main_branch:add(inception(576, {{  0},{128,192},{192,256},{'max', 0}}, bn)) -- 4(e)
  main_branch:add(cudnn.SpatialConvolution(1024,1024,2,2,2,2))

  -- to del
  th.addSBN(main_branch, 1024, bn)

  main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}}, bn)) -- 5(a)
  main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}}, bn)) -- 5(b)
  main_branch:add(cudnn.SpatialAveragePooling(7,7,1,1))
  main_branch:add(nn.View(1024):setNumInputDims(3))
  main_branch:add(nn.Linear(1024, nC))
  main_branch:add(nn.LogSoftMax())

  local model = nn.Sequential():add(features):add(main_branch)

  -- init
  th.iniMod(model, ini)

  return model, {{}}
end

----------------------------------------------------------------------
-- Create alexnet model for fine-tuning.
--
-- Input
--   nC     -  #classes
--   bn     -  type of BN
--   ini    -  initialize method
--
-- Output
--   model  -  pre-trained model
--   mods   -  sub-modules needed to re-train, m x
function goo.newT(nC, bn, ini)
  local model = torch.load(modPath0)

  -- remove last fully connected layer
  local mod0 = model.modules[2].modules[8]
  model.modules[2]:remove(8)

  -- insert a new one
  local mod = nn.Linear(1024, nC)
  model.modules[2]:insert(mod, 8)

  -- init
  th.iniMod(mod, ini)

  return model, {{mod}}
end

----------------------------------------------------------------------
-- Create goonet model for fine-tuning.
--
-- In: m images
-- Out: nC x softmax
--
-- Input
--   nC     -  #classes
--   ini    -  initialize method
--   m      -  #model
--
-- Output
--   model  -  pre-trained model
--   mods   -  sub-modules needed to re-train, m x
function goo.newStnClfy(nC, ini, m)
  -- alex net
  local model = nn.Sequential()

  -- feature extraction
  local gooNets = nn.ParallelTable()
  model:add(gooNets)
  for i = 1, m do
    local gooNet = torch.load(modPath0)
    gooNets:add(gooNet)

    -- remove last fully connected layer
    gooNet.modules[2]:remove(9)
    gooNet.modules[2]:remove(8)
  end

  -- concate the output
  model:add(nn.JoinTable(2))

  -- insert a new last layer
  local mod = nn.Linear(1024 * m, nC)
  model:add(mod)
  model:add(nn.LogSoftMax())

  -- init
  th.iniMod(mod, ini)

  return model, {mod}
end

----------------------------------------------------------------------
-- Create the localization net for STN.
--
-- Input
--   bn     -  type of BN
--   ini    -  init method
--   loc    -  localization network
--   k      -  #dimension
--
-- Output
--   model  -  model
--   mods   -  sub-modules needed to re-train, m x
function goo.newStnLoc(bn, ini, loc, k)
  assert(bn)
  assert(ini)
  assert(loc)
  assert(k)

  -- load old model
  local model = torch.load(modPath0)
  local mod1, mod2

  if loc == 'type1' then
    local main = model.modules[2]

    -- remove the classifier layer
    main:remove(9)
    main:remove(8)
    main:remove(7)
    main:remove(6)

    -- add a new 1x1 convolutional layer
    mod1 = cudnn.SpatialConvolution(1024, 128, 1, 1, 1, 1)
    main:add(mod1)
    th.addSBN(main, 128, bn)
    main:add(cudnn.ReLU(true))

    -- add a fully-connected layer
    main:add(nn.View(128 * 7 * 7))
    mod2 = nn.Linear(128 * 7 * 7, k)
    main:add(mod2)
    th.addBN(main, k, bn)
    main:add(cudnn.ReLU(true))

    -- init
    th.iniMod(mod1, ini)
    th.iniMod(mod2, ini)

  else
    assert(nil, string.format('unknown loc: %s', loc))
  end

  return model, {mod1, mod2}
end

return goo
