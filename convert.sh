./ocr --build_precision="fp16" --det_onnx_dir="../myModels/det.onnx" --rec_onnx_dir="../myModels/ch_rec_v3.onnx" --save_engine_dir="../myEngines/" --rec_char_dict_path="../myModels/dict_txt/output.txt" --rec_batch_num=1 --det=true --cls=false --rec=true --image_dir="../testImgs/11.jpg" --output="./output/"