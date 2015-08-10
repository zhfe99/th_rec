#!/usr/bin/env th
-- Generate lmdb.
--
-- History
--   create  -  Feng Zhou (zhfe99@gmail.com), 08-03-2015
--   modify  -  Feng Zhou (zhfe99@gmail.com), 08-09-2015

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
local dat = ThDat(dbe, ver)
local PATH = dat.PATH

-- config
local confPath = string.format('Models/%s_%s_conf.lua', dbe, ver)
local config = paths.dofile(confPath)

local TrainingFiles = FileSearcher {
  Name = 'TrainingFilenames',
  CachePrefix = PATH.trCach,
  -- CachePrefix = config.TRAINING_DIR,
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
  CacheFiles = true,
  PathList = {config.VALIDATION_PATH},
  SubFolders = true,
  MaxFilenameLength = 200
}

local TrainDB = lmdb.env {
  Path = PATH.trLmdb,
  Name = 'TrainDB'
}

local ValDB = lmdb.env {
  Path = PATH.teLmdb,
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
    im = im:reshape(1, im:size(1), im:size(2))
  end
  if im:size(1) == 1 then
    im = torch.repeatTensor(im, 3, 1, 1)
  end
  if im:size(1) > 3 then
    im = im[{{1, 3}, {}, {}}]
  end
  return im
end

----------------------------------------------------------------------
-- Load image data.
--
-- Input
--   filename  -  file path
--
-- Output
--   img       -  image
local LoadImgData = function(filename)
  local ok, img0 = pcall(gm.Image, filename)

  -- error
  if not ok or img0 == nil then
    print('Image is buggy')
    print(filename)
    os.exit()
  end

  -- pre-process
  img0 = img0:toTensor('byte', 'RGB', 'DHW')
  local img = PreProcess(img0)
  return img, img0
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
  local me = {0, 0, 0}
  local std = {0, 0, 0}
  local nImg = charTensor:size(1)
  for i = 1, nImg do
    local filename = ffi.string(torch.data(charTensor[i]))
    local img, img0 = LoadImgData(filename)

    -- update mean & std
    for j = 1, 3 do
      me[j] = me[j] + img[j]:float():mean()
      std[j] = std[j] + img[j]:float():std()
    end

    require 'image'
    local img2 = image.load(filename)
    image.save(string.format('tmp_%d.jpg', 0), img0:float() / 255)
    image.save(string.format('tmp_%d.jpg', 1), img:float() / 255)
    image.save(string.format('tmp_%d.jpg', 2), img2)

    -- local img00 = img0:float() / 255
    local img0b = image.compressJPG(img0:float() / 255)
    local img0c = image.decompressJPG(img0b)
    image.save(string.format('tmp_%dc.jpg', 0), img0c)

    local debugger = require('fb.debugger')
    debugger.enter()

    local data = {Data = img, Name = NameFile(filename)}
    cursor:put(config.Key(i), data, lmdb.C.MDB_NODUPDATA)
    if i % 1000 == 0 then
      txn:commit()
      print(env:stat())
      collectgarbage()
      txn = env:txn()
      cursor = txn:cursor()
    end
    xlua.progress(i, nImg)
  end
  txn:commit()
  env:close()

  -- update mean & std
  for j = 1, 3 do
    me[j] = me[j] / nImg
    std[j] = std[j] / nImg
  end

  print(string.format('mean: %.3f, %.3f, %.3f', me[1], me[2], me[3]))
  print(string.format('std : %.3f, %.3f, %.3f', std[1], std[2], std[3]))
end

TrainingFiles:ShuffleItems()
LMDBFromFilenames(ValidationFiles.Data, ValDB)
LMDBFromFilenames(TrainingFiles.Data, TrainDB)
