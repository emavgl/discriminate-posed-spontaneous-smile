# Frontalization

## Extract original landmarks

```
cd frontalized frontalization-1
python3 extract_frames.py dataset
bash extract_landmark.sh dataset
bash clean_nofront.sh dataset
```

**landmarks\_orig\_only** will contain the extracted landmarks (no frontalized)

## Frontalization-1
Use https://github.com/dougsouza/face-frontalization
Omitted. We don't use this anymore because of bad results.

## Frontalization-2
O.C. Hamsici, P.F.U. Gotardo, A.M. Martinez, “Learning Spatially-Smooth Mappings in Non-Rigid Structure from Motion”, European Conference on Computer Vision, ECCV 2012, Firenze, Italy.

### How to use

Convert landmarks into mat files
```
python lmToMat.py ./1\ Frontalization/frontalization-2/landmarks/landmarks
```

Run matlab script
```
cd ./1\ Frontalization/frontalization-2/frontal_face_estimation/functions
matlab -nosplash -nodesktop -r automation
cd ../../../..
```

Convert back frontalized .mat files into .lm
```
python matToLm.py ./1\ Frontalization/frontalization-2/landmarks/alignedLandmarks
```
