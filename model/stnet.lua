#!/usr/bin/env th
-- STN network.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-06-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-18-2015

require 'stn'

local stn = {}

----------------------------------------------------------------------
-- Create the stn model.
--
-- Input
--   locnet  -  localization network
--   nC      -  #classes
--   nGpu    -  #gpus
--   isBn    -  flag of using BN, true | false
--   iniAlg  -  init method
--   inSiz   -  input size
--
-- Output
--   model   -  model
function stn.new(locnet, isBn, inSiz)
  local spanet = nn.Sequential()
  local concat = nn.ConcatTable()

  -- first branch is there to transpose inputs to BHWD, for the bilinear sampler
  local tranet = nn.Sequential()
  tranet:add(nn.Identity())
  tranet:add(nn.Transpose({2, 3}, {3, 4}))

  -- full model
  -- local outLayer = nn.Linear(128, 6)
  -- outLayer.weight:fill(0)
  -- local bias = torch.FloatTensor(6):fill(0)
  -- bias[1] = 1
  -- bias[5] = 1
  -- outLayer.bias:copy(bias)
  -- locnet:add(outLayer)
  -- locnet:add(nn.View(2, 3))

  -- scale + translation
  local outLayer = nn.Linear(128, 3)
  outLayer.weight:fill(0)
  local bias = torch.FloatTensor(3):fill(0)
  bias[1] = 1
  bias[2] = 0
  bias[3] = 0
  outLayer.bias:copy(bias)
  locnet:add(outLayer)
  locnet:add(nn.AffineTransformMatrixGenerator(false, true, true))

  -- grid generation
  locnet:add(nn.AffineGridGeneratorBHWD(inSiz, inSiz))

  -- we need a table input for the bilinear sampler, so we use concattable
  concat:add(tranet)
  concat:add(locnet)

  spanet:add(concat)
  spanet:add(nn.BilinearSamplerBHWD())

  -- and we transpose back to standard BDHW format for subsequent processing by nn modules
  spanet:add(nn.Transpose({3, 4}, {2, 3}))

  return spanet, {outLayer}
end

return stn
