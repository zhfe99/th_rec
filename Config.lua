local ImageNetClasses = torch.load('./ImageNetClasses')
for i=1001,#ImageNetClasses.ClassName do
  ImageNetClasses.ClassName[i] = nil
end

function Key(num)
  return string.format('%07d', num)
end

return {
  TRAINING_PATH = '/home/ma/feng/data/imgnet/v2/train/',
  VALIDATION_PATH = '/home/ma/feng/data/imgnet/v2/val/',
  VALIDATION_DIR = '/home/ma/feng/save/imgnet/torch/data/imgnet_v2_val/',
  TRAINING_DIR = '/home/ma/feng/save/imgnet/torch/data/imgnet_v2_train/',
  ImageSize = 256,
  SampleSize = {3, 224, 224},
  ValidationLabels = torch.load('./ValidationLabels'),
  ImageNetClasses = ImageNetClasses,
  DataMean = 118.380948,
  DataStd = 61.896913,
  Compressed = true,
  Key = Key
}
