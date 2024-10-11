AUO Crack Segmentation C# API


1. 檔案構成:
	1. OnnxInference class DLL file: 
		- AUOCrackOnnxInference.dll
		- Onnx 處理Class，包含LoadModel/Inference/Output等API
		
	2. 測試程式(C#):
		- Program.cs
		- 所有API使用皆可參考此份檔案

	3. Readme.txt

2. 相依性:
	我們使用Visual Studio 2022開發，以下為我們有額外安裝的套件:
	1. Microsoft.ML.OnnxRuntime.Gpu: 1.18.1 (因我們的onnx model IR version為10，無法使用更低版本的onnxruntime)
	2. OpenCvSharp4.Windows: 4.8.0.20230708


3. 測試程式可調參數(Program.cs)
	1. now_device: 指定運行GPU(目前只接受單一GPU)

	2. HEAD_TYPE: "ms"/"linear"，目前版本不可調整為linear

	3. is_fp16: True/False，使用FP16的模型(C# 無FP32轉換為FP16模型功能)

	4. onnx_model_path: onnx模型路徑或根據HEAD_TYPE,HEAD_SCALE_COUNT,is_fp16資訊，透過generate_onnx_model_path自動轉換為對應模型路徑

	5. image_path: 需要推理的圖片，可接受單一圖像路徑/目錄路徑(目錄下全部圖片作為input)/列表(所有需要推理的圖片路徑列表)

	6. batch_size: 推理批次大小，可為1

	7. seg_area_threshold: 過濾所有面積大小小於seg_area_threshold的雜訊區域。

	8. is_dilation: true/false，是否要開啟針對破碎的crack區域進行膨脹的功能(開啟此功能後crack形狀較平滑，但誤報率提高)
	
	9. return_logits: true/false，Output API 是否回傳segmentation mask (影像像素數值 0為background, 1為crack)

	10. show_result: true/false，是否要開啟視覺化功能，只有在return_logits=true有效(影像色彩紅色為crack，並含有crack區域的綠色bbox) 

4. Program.cs 功能介紹:
	- 從參數image_path 取得所有圖片路徑
	List<string> image_list = GetImageList(string image_path);

	- 從圖片路徑列表中讀取圖片
	List<Mat> ori_images = LoadImages(List<string> image_list);

	- 圖片前處理(包含將圖片轉換為Float[])
	float[] preprocessedImages = PreprocessImages(ori_images, new Size(644, 644));

	- 視覺化，開啟結果視窗並將segmentation結果和BBOX畫在原圖上
	visualize(outputMats, ori_images, image_list, predicts, bboxes);

5. OnnxInference.dll API 功能介紹:
	- 加載模型
		var onnxInference = new OnnxInference(); 
		onnxInference.LoadModel(onnx_model_path, HEAD_TYPE, now_device, is_fp16, batch_size, warm_up); 
	 	- warm_up 次數預設為4,可設為0以關閉此功能

	- 推理圖片
		- 使用FP32模型
		outputMats = onnxInference.Inference(preprocessedImages, batch_size);
		
		- 使用FP16模型
		使用前必須將前處理後的Float data 轉換為 Float16 格式
		outputMats = onnxInference.InferenceFP16(preprocessedImagesFP16, batch_size);

	- 處理結果輸出
		-  Predicts: 所有圖片的是否為Crack結果列表，
			[
				"OK", // img0 predict result
				"OK",
				"NG", // img2 predict crack!
				...
			]
		-  bboxes: Crack區域的外接四邊形座標列表 
			[
				// img0 
				[
					[x1, y1, x2, y2], //img0 crack0 coordinate
					[x1, y1, x2, y2], //img0 crack1 coordinate
					...
				], 
				// img1 
				[
					[x1, y1, x2, y2], //img1 crack0 coordinate
					[x1, y1, x2, y2], //img1 crack1 coordinate
				],...
			]
		-  logits: segmentation result列表(0:bg ,1:crack)，只有在return_logits=true有效

		var result = onnxInference.Output(outputMats, ori_images, seg_area_threshold, is_fp16, is_dilation, return_logits);
		List<string> predicts = result.Item1; 
		List<List<Rect>> bboxes = result.Item2; 
		List<Mat> logits = result.Item3; 

