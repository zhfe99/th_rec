#!/usr/bin/env th
-- Generate lmdb.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015

require 'eladtools'
require 'image'
require 'xlua'
require 'lmdb'
local gm = require 'graphicsmagick'
local ffi = require 'ffi'
local th = require 'lua_th'

-- argument
cmd = torch.CmdLine()
cmd:option('-dbe', 'car', 'database name')
cmd:option('-ver', 'v1c', 'version')
local params = cmd:parse(arg)

-- data
local dbe = params.dbe
local ver = params.ver
dat = ThDat(dbe, ver)

-- config
confPath = string.format('Models/Config_%s_%s', dbe, ver)
local config = require(confPath)

local debugger = require('fb.debugger')
debugger.enter()

local TrainingFiles = FileSearcher {
  Name = 'TrainingFilenames',
  CachePrefix = config.TRAINING_DIR,
  MaxNumItems = 1e8,
  CacheFiles = true,
  PathList = {config.TRAINING_PATH},
  SubFolders = true,
  MaxFilenameLength = 200
}

local ValidationFiles = FileSearcher {
  Name = 'ValidationFilenames',
  CachePrefix = config.VALIDATION_DIR,
  MaxNumItems = 1e8,
  PathList = {config.VALIDATION_PATH},
  SubFolders = true,
  MaxFilenameLength = 200
}

local TrainDB = lmdb.env {
  Path = config.TRAINING_DIR,
  Name = 'TrainDB'
}

local ValDB = lmdb.env {
  Path = config.VALIDATION_DIR,
  Name = 'ValDB'
}

----------------------------------------------------------------------
-- Pre-process image.
--
-- Input
--   Img  -  original image
--
-- Output
--   img  -  new img
local PreProcess = function(Img)
  --minimum side of ImageSize
  local im = image.scale(Img, '^' .. config.ImageSize)

  if im:dim() == 2 then
    im = im:reshape(1,im:size(1), im:size(2))
  end
  if im:size(1) == 1 then
    im=torch.repeatTensor(im, 3, 1, 1)
  end
  if im:size(1) > 3 then
    im = im[{{1,3},{},{}}]
  end
  return im
end

----------------------------------------------------------------------
-- Load image data.
--
-- Input
--   filename  -  file path
local LoadImgData = function(filename)
  local ok, img = pcall(gm.Image, filename)
  if not ok or img == nil then
    print('Image is buggy')
    print(filename)
    local debugger = require('fb.debugger')
    debugger.enter()
    os.exit()
  end
  img = img:toTensor('float', 'RGB', 'DHW')
  img = PreProcess(img)
  if config.Compressed then
    return image.compressJPG(img)
  else
    return img
  end
end

----------------------------------------------------------------------
-- Rename the file.
--
-- Input
--   filename  -  file path
--
-- Output
--   name      -  key used in lmdb
function NameFile(filename)
  local ext = paths.extname(filename)
  local foldPath = paths.dirname(filename)
  local parts = string.split(foldPath, '/')
  local foldNm = parts[#parts]
  local imgNm = paths.basename(filename, ext)
  local name = string.format('%s/%s', foldNm, imgNm)
  return name
end

----------------------------------------------------------------------
-- Generate lmdb from a list of files.
--
-- Input
--   charTensor  -  file name list
--   env         -  lmdb env
function LMDBFromFilenames(charTensor, env)
  env:open()
  local txn = env:txn()
  local cursor = txn:cursor()
  for i=1,charTensor:size(1) do
    local filename = ffi.string(torch.data(charTensor[i]))
    local data = {Data = LoadImgData(filename), Name = NameFile(filename)}

    cursor:put(config.Key(i), data, lmdb.C.MDB_NODUPDATA)
    if i % 1000 == 0 then
      txn:commit()
      print(env:stat())
      collectgarbage()
      txn = env:txn()
      cursor = txn:cursor()
    end
    xlua.progress(i, charTensor:size(1))
  end
  txn:commit()
  env:close()
end

TrainingFiles:ShuffleItems()
LMDBFromFilenames(TrainingFiles.Data, TrainDB)
LMDBFromFilenames(ValidationFiles.Data, ValDB)
