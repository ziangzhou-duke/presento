############ install openface ############
pushd pose

python setup.py install
./models/get-models.sh

popd
