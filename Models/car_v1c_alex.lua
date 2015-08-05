paths.dofile('alex.lua')

local model = newModel(196, 1, false, 'xavier')

local loss = nn.ClassNLLCriterion()

local nEpo = 60

local nEpoSv = 20

local lrs = {
  { 1,   18, 1e-2, 5e-4},
  {19,   29, 5e-3, 5e-4},
  {30,   43, 1e-3, 0},
  {44,   52, 5e-4, 0},
  {53, nEpo, 1e-4, 0},
}

local batchSiz = 128

local mom = 0.9

return model, loss, nEpo, nEpoSv, batchSiz, lrs, mom
