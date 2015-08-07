#!/usr/bin/env th
-- STN network.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-06-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-07-2015

local stn = {}

----------------------------------------------------------------------
-- Create the stn model.
--
-- Input
--   locNet  -  localization network
--   nC      -  #classes
--   nGpu    -  #gpus
--   isBn    -  flag of using BN, true | false
--   iniAlg  -  init method
--   inSiz   -  input size
--
-- Output
--   model   -  model
function stn.new(locNet, nC, nGpu, isBn, iniAlg, inSiz)
  require 'stn'
  local spanet = nn.Sequential()
  local concat = nn.ConcatTable()

  -- first branch is there to transpose inputs to BHWD, for the bilinear sampler
  local tranet = nn.Sequential()
  tranet:add(nn.Identity())
  tranet:add(nn.Transpose({2, 3}, {3, 4}))

  -- second branch is the localization network
  local locnet;
  if locNet == 'alex' then
    local alex = require 'Models.alex'
    locnet = alex.newStnTrun(nC, nGpu, isBn, iniAlg)
  end
  -- = nn.Sequential()
  -- locnet:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  -- locnet:add(cudnn.SpatialConvolution(1, 20, 5, 5))
  -- locnet:add(cudnn.ReLU(true))
  -- locnet:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
  -- locnet:add(cudnn.SpatialConvolution(20, 20, 5, 5))
  -- locnet:add(cudnn.ReLU(true))
  -- locnet:add(nn.View(20 * 2 * 2))
  -- locnet:add(nn.Linear(20 * 2 * 2, 20))
  -- locnet:add(cudnn.ReLU(true))

  -- we initialize the output layer so it gives the identity transform
  local outLayer = nn.Linear(128, 6)
  outLayer.weight:fill(0)
  local bias = torch.FloatTensor(6):fill(0)
  bias[1] = 1
  bias[5] = 1
  outLayer.bias:copy(bias)
  locnet:add(outLayer)

  -- there we generate the grids
  locnet:add(nn.View(2, 3))
  locnet:add(nn.AffineGridGeneratorBHWD(inSiz, inSiz))

  -- we need a table input for the bilinear sampler, so we use concattable
  concat:add(tranet)
  concat:add(locnet)

  spanet:add(concat)
  spanet:add(nn.BilinearSamplerBHWD())

  -- and we transpose back to standard BDHW format for subsequent processing by nn modules
  spanet:add(nn.Transpose({3, 4}, {2, 3}))

  return spanet
end

return stn
