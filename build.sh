# clean TransformerEngine
/usr/local/bin/pip uninstall transformer_engine -y
rm -rf build 
rm -rf transformer_engine.egg-info
rm -f transformer_engine/transformer_engine_torch.cpython-310-x86_64-linux-gnu.so
rm -f libtransformer_engine.so
rm -f transformer_engine_torch.cpython-310-x86_64-linux-gnu.so

# install TransformerEngine
export MUSA_HOME=/usr/local/musa
export NVTE_FRAMEWORK=musa
/usr/local/bin/pip wheel --no-build-isolation -v -w ./dist . 2>&1 | tee build.log
