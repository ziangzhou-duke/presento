############ install openface ############
pushd pose/openface

python setup.py install
./models/get-models.sh

popd

############ install openpose ############
pushd pose/openpose

pip install -r requirements.txt

popd
