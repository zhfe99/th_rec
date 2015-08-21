#!/usr/bin/env th
-- STN network.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-06-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-21-2015

require 'stn'

local stn = {}

----------------------------------------------------------------------
-- Create the stn model.
--
-- Input
--   locnet  -  localization network
--              the output layer should be, nBat x k
--   isBn    -  flag of using BN, true | false
--   tran    -  transformation name, 'aff' | 'sca' | 'tra' | 'rot' | 'tra2' | 'tras'
--   k       -  #dimension of locnet
--   inSiz   -  input size
--
-- Output
--   model   -  model
--   mods    -  module needed to be update, m x
function stn.new(locnet, isBn, tran, k, inSiz)
  local spanet = nn.Sequential()
  local concat = nn.ConcatTable()

  -- first branch is there to transpose inputs to BHWD, for the bilinear sampler
  local tranet = nn.Sequential()
  tranet:add(nn.Identity())
  tranet:add(nn.Transpose({2, 3}, {3, 4}))

  -- out layer
  local outLayer
  if tran == 'aff' then
    -- affine
    outLayer = nn.Linear(k, 6)

    -- init
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(6):fill(0)
    bias[1] = 1
    bias[5] = 1
    outLayer.bias:copy(bias)

    locnet:add(outLayer)
    locnet:add(nn.View(2, 3))

  elseif tran == 'tra' then
    -- translation
    outLayer = nn.Linear(k, 2)

    -- init
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(2):fill(0)
    bias[1] = 0
    bias[2] = 0
    outLayer.bias:copy(bias)

    locnet:add(outLayer)
    locnet:add(nn.AffineTransformMatrixGenerator(false, false, true))

  elseif tran == 'tras2' then
    -- translation with fixed half-scaling
    outLayer = nn.Linear(k, 2)

    -- init
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(2):fill(0)
    bias[1] = 0
    bias[2] = 0
    outLayer.bias:copy(bias)

    locnet:add(outLayer)
    locnet:add(nn.AffineTransformMatrixGenerator(false, false, true, .5))

  elseif tran == 'tras' then
    -- translation + scaling
    outLayer = nn.Linear(k, 3)

    -- init
    outLayer.weight:fill(0)
    local bias = torch.FloatTensor(3):fill(0)
    bias[1] = 1
    bias[2] = 0
    bias[3] = 0
    outLayer.bias:copy(bias)

    locnet:add(outLayer)
    locnet:add(nn.AffineTransformMatrixGenerator(false, true, true))

  else
    assert(nil, string.format('unknown tran: %s', tran))
  end

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
