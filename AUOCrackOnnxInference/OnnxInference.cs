using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System;
using System.Linq;
using System.IO;
using System.Collections.Generic;

namespace AUOCrackOnnxInference
{
    public class OnnxInference
    {
        private InferenceSession ort_sess;
        private const string InputNodeName = "input1";

        private Size pad_size;
        private int rgb_img_length;
        private int bk_img_length;
        private int HEAD_SCALE_COUNT = 1;


        public string GenerateOnnxModelPath(string head_type, bool is_fp16)
        {
            string onnx_model_path = @"D:\AUOCrackCSAPI\AUOCrackCSAPI\onnx_model\dinov2b14.onnx";

            onnx_model_path = onnx_model_path.Replace(".onnx", $"_{head_type}.onnx");

            // Modify the path if head_type is "ms"
            if (head_type == "ms")
            {
                onnx_model_path = onnx_model_path.Replace(".onnx", $"_{HEAD_SCALE_COUNT}.onnx");
            }
            // Check if fp16 and update path if necessary
            if (is_fp16)
            {
                string fp16_path = onnx_model_path;
                string ori_model_path = onnx_model_path;

                // Check if the model is already FP16
                if (!onnx_model_path.Contains("_fp16"))
                {
                    fp16_path = onnx_model_path.Replace(".onnx", "_fp16.onnx");
                    onnx_model_path = fp16_path;
                }
                else
                {
                    ori_model_path = onnx_model_path.Replace("_fp16.onnx", ".onnx");
                }

                // Check if the FP16 model exists
                if (!File.Exists(fp16_path))
                {
                    Console.WriteLine("FP16 model not exists! C# can not Generating FP16 model!");
                }
            }

            return onnx_model_path;
        }

        public void LoadModel(string onnx_model_path, string HEAD_TYPE, int now_device, bool is_fp16, int batch_size = 8, int warm_up = 4)
        {
            if (HEAD_TYPE == "ms")
            {
                this.pad_size = new Size(644, 644);
            }
            else if (HEAD_TYPE == "linear")
            {
                this.pad_size = new Size(518, 518);
            }

            this.bk_img_length = pad_size.Height * pad_size.Width;
            this.rgb_img_length = 3 * pad_size.Height * pad_size.Width;


            if (onnx_model_path == "")
            {
                onnx_model_path = GenerateOnnxModelPath(HEAD_TYPE, is_fp16);
            }
            else
            {
                // Check if the model exists
                if (!File.Exists(onnx_model_path))
                {
                    Console.WriteLine("Onnx model not exists!");
                }
            }

            var options = new SessionOptions();
            options.AppendExecutionProvider_CUDA(now_device);
            //options.AppendExecutionProvider_CPU(0);
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            ort_sess = new InferenceSession(onnx_model_path, options);

            Console.WriteLine($"Warm up {warm_up} times.");
            float[] dummy = new float[batch_size*rgb_img_length];
            for (int i = 0; i < warm_up; i++)
            {
                if (is_fp16)
                {
                    Float16[] dummyFP16 = dummy.Select(y => (Float16)y).ToArray(); // fp32 to fp16 preprocessedImages
                    InferenceFP16(dummyFP16, batch_size);
                }
                else
                {
                    Inference(dummy, batch_size);
                }
            }
        }


        public static Mat MorphologicalOperations(Mat mask)
        {
            // 定義膨脹核（結構元素）和侵蝕核
            Mat dkernel = Cv2.GetStructuringElement(MorphShapes.Cross, new Size(10, 10));
            Mat ekernel = Mat.Ones(new Size(10, 10), MatType.CV_8U); // Ones matrix for erosion kernel

            // 進行膨脹操作，iterations=4
            Cv2.Dilate(mask, mask, dkernel, iterations: 4);

            // 進行侵蝕操作，iterations=2
            Cv2.Erode(mask, mask, ekernel, iterations: 2);

            return mask;
        }

        public (List<string>, List<List<Rect>>, List<Mat>) Output(List<Mat> outputMats, List<Mat> ori_images, int seg_area_threshold, bool is_fp16, bool is_dilation, bool return_logits)
        {
            // Assuming your model output is a tensor
            //Mat outputMat;
            string predict_str;
            List<string> Predicts = new List<string>();
            List<List<Rect>> Bboxes = new List<List<Rect>>();

            //for (int i = 0; i < outputMats.Count; i++)
            foreach (Mat outputMat in outputMats)
            {
                //outputMat = outputMats[i];
                if (is_dilation)
                {
                    MorphologicalOperations(outputMat);
                }

                Cv2.FindContours(outputMat, out Point[][] infer_contours, out HierarchyIndex[] hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

                List<Rect> nowRects = new List<Rect>();

                if (seg_area_threshold <= 0)
                {
                    foreach (var contour in infer_contours)
                    {
                        nowRects.Add(Cv2.BoundingRect(contour)); // calculate the bounding box
                    }
                }
                else
                {
                    foreach (var contour in infer_contours)
                    {
                        double area = Cv2.ContourArea(contour);

                        // Filter areas smaller than "seg_area_threshold"
                        if (area <= seg_area_threshold)
                        {
                            if (return_logits)
                            {
                                // Fill the small area cracks to bg
                                Cv2.DrawContours(outputMat, new[] { contour }, -1, Scalar.All(0), thickness: Cv2.FILLED);
                            }
                        }
                        else
                        {
                            nowRects.Add(Cv2.BoundingRect(contour)); // calculate the bounding box
                        }
                    }
                }

                predict_str = nowRects.Count > 0 ? "NG" : "OK";
                //Console.WriteLine($"Predict: {predict_str}, Area: {nowRects.Count}");

                Predicts.Add(predict_str);
                Bboxes.Add(nowRects);
            }

            return return_logits ? (Predicts, Bboxes, outputMats) : (Predicts, Bboxes, null);
        }


        public List<Mat> InferenceFP16(Float16[] preprocessedImages, int batch_size = 8)
        {
            long start_time = 0; // timer
            long total_time = 0; // timer
            var watch = System.Diagnostics.Stopwatch.StartNew(); // timer

            List<Mat> outputMats = new List<Mat>();

            int images_count = preprocessedImages.Length / rgb_img_length;

            int num_batches = (int)Math.Ceiling((float)images_count / batch_size);

            for (int batch_idx = 0; batch_idx < num_batches; batch_idx++)
            {
                start_time = watch.ElapsedMilliseconds; // timer
                int start_idx = batch_idx * batch_size;
                int end_idx = Math.Min((batch_idx + 1) * batch_size, images_count);
                int now_batch_size = end_idx - start_idx;

                Float16[] inputData = new Float16[now_batch_size * rgb_img_length];
                var inputTensor = new DenseTensor<Float16>(inputData, new[] { now_batch_size, 3, pad_size.Height, pad_size.Width });

                // Copy preprocessed image data into the batch tensor
                Array.Copy(preprocessedImages, start_idx * rgb_img_length, inputData, 0, now_batch_size * rgb_img_length);

                // prepare input data
                var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(InputNodeName, inputTensor) };

                // Run inference
                var results = ort_sess.Run(inputs);

                // Convert results to a list ????
                var resultList = results.ToList();

                // result to list of Mat
                var outputTensor = resultList[0].AsEnumerable<long>().ToArray();
                byte[] byteArray = Array.ConvertAll(outputTensor, x => (byte)x);  // long(int64) -> byte
                for (int i = 0; i < now_batch_size; i++)
                {
                    byte[] result = byteArray.Skip(i * bk_img_length).Take(bk_img_length).ToArray(); // get each inference result
                    Mat outputMat = new Mat(pad_size.Height, pad_size.Width, MatType.CV_8UC1, result);
                    outputMats.Add(outputMat);
                }

                total_time += watch.ElapsedMilliseconds - start_time; // timer
            }

            Console.WriteLine($"Inference {images_count} images: {total_time} ms, avg inference time: {total_time / images_count} ms"); // timer
            watch.Stop(); // timer

            return outputMats;
        }


        public List<Mat> Inference(float[] preprocessedImages, int batch_size = 8)
        {
            long start_time = 0; // timer
            long total_time = 0; // timer
            var watch = System.Diagnostics.Stopwatch.StartNew(); // timer

            List<Mat> outputMats = new List<Mat>();

            int images_count = preprocessedImages.Length / rgb_img_length;

            int num_batches = (int)Math.Ceiling((float)images_count / batch_size);

            for (int batch_idx = 0; batch_idx < num_batches; batch_idx++)
            {
                start_time = watch.ElapsedMilliseconds; // timer
                int start_idx = batch_idx * batch_size;
                int end_idx = Math.Min((batch_idx + 1) * batch_size, images_count);
                int now_batch_size = end_idx - start_idx;

                float[] inputData = new float[now_batch_size * rgb_img_length];
                var inputTensor = new DenseTensor<float>(inputData, new[] { now_batch_size, 3, pad_size.Height, pad_size.Width });

                // Copy preprocessed image data into the batch tensor
                Array.Copy(preprocessedImages, start_idx * rgb_img_length, inputData, 0, now_batch_size * rgb_img_length);

                // prepare input data
                var inputs = new NamedOnnxValue[] { NamedOnnxValue.CreateFromTensor(InputNodeName, inputTensor) };

                // Run inference
                var results = ort_sess.Run(inputs);

                // Convert results to a list ????
                var resultList = results.ToList();

                // result to list of Mat
                var outputTensor = resultList[0].AsEnumerable<long>().ToArray();
                byte[] byteArray = Array.ConvertAll(outputTensor, x => (byte)x);  // long(int64) -> byte
                for (int i = 0; i < now_batch_size; i++)
                {
                    byte[] result = byteArray.Skip(i * bk_img_length).Take(bk_img_length).ToArray(); // get each inference result
                    Mat outputMat = new Mat(pad_size.Height, pad_size.Width, MatType.CV_8UC1, result);
                    outputMats.Add(outputMat);
                }

                total_time += watch.ElapsedMilliseconds - start_time; // timer
            }

            Console.WriteLine($"Inference {images_count} images: {total_time} ms, avg inference time: {total_time / images_count} ms"); // timer
            watch.Stop(); // timer

            return outputMats;
        }
    }
}
