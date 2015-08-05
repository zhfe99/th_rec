#!/usr/bin/env th
-- car v1c googlenet.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-05-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-05-2015

paths.dofile('goo.lua')

local model = newModel(196, 1, false)

local loss = nn.ClassNLLCriterion()

local nEpo = 60

local lrs = {
  { 1,   18, 1e-2, 5e-4},
  {19,   29, 5e-3, 5e-4},
  {30,   43, 1e-3, 0},
  {44,   52, 5e-4, 0},
  {53, nEpo, 1e-4, 0},
}

local batchSiz = 100

local mom = 0.9

return model, loss, nEpo, nEpoSv, batchSiz, lrs, mom
