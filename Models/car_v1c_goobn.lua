paths.dofile('goo.lua')

local nC = 196

local model = newModel(nC, opts.nGpu, true, 'xavier')

local loss = nn.ClassNLLCriterion()

local nEpo = 120

local nEpoSv = 40

local lrs = {
  {  1,   20, 1e-2, 5e-4},
  { 21,   40, 5e-3, 5e-4},
  { 41,   60, 1e-3, 5e-4},
  { 61,   80, 5e-4, 5e-4},
  { 81,  100, 1e-4, 5e-4},
  {101, nEpo, 5e-5, 5e-4},
}

local batchSiz = 50

local mom = 0.9

return model, loss, nEpo, nEpoSv, batchSiz, lrs, mom
