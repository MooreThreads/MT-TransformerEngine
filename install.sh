export NVTE_FRAMEWORK=musa
pip uninstall transformer_engine -y
rm -rf build 
rm -rf transformer_engine.egg-info
rm -rf transformer_engine/transformer_engine_torch.cpython-310-x86_64-linux-gnu.so
python setup.py develop > develop.log 2>&1
