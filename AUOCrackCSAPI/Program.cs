using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using AUOCrackOnnxInference;

class Program
{
    static List<string> GetImageList(object imagePath)
    {
        List<string> imageList = new List<string>();

        if (imagePath is List<string> list) // If imagePath is already a list
        {
            imageList = list;
        }
        else if (imagePath is string path) // If imagePath is a string (file or directory)
        {
            if (File.Exists(path)) // Check if it's a file
            {
                imageList.Add(path);
            }
            else if (Directory.Exists(path)) // Check if it's a directory
            {
                var files = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories) // Get all files in subdirectories
                                      .Where(file => IsImageFile(file)) // Filter only image files
                                      .OrderBy(file => file) // Sort the files
                                      .ToList();

                imageList.AddRange(files);
            }
            else
            {
                Console.WriteLine($"Provided image path {imagePath} is not valid.");
            }
        }
        else
        {
            Console.WriteLine($"Provided image path {imagePath} is not valid.");
        }

        return imageList;
    }

    static bool IsImageFile(string filePath)
    {
        string[] validExtensions = { ".jpg", ".jpeg", ".png", ".bmp" };
        string extension = Path.GetExtension(filePath).ToLower();
        return validExtensions.Contains(extension);
    }

    static Mat Padding(Mat img, Size pad_size)
    {
        // Check if the image is larger than the target size
        if (img.Rows > pad_size.Height || img.Cols > pad_size.Width)
        {
            Console.WriteLine("img bigger than padding size");
            return img;
        }

        // If the image already matches the target size just return it
        if (img.Rows == pad_size.Height && img.Cols == pad_size.Width)
        {
            return img;
        }

        // Create a new Mat filled with zeros (black padding) with the target size and 3 channels
        Mat paddingResult = new Mat(pad_size.Height, pad_size.Width, MatType.CV_8UC3, new Scalar(0, 0, 0));

        // Copy the original image into the top-left corner of the paddingResult
        img.CopyTo(paddingResult[new Rect(0, 0, img.Cols, img.Rows)]);

        return paddingResult;
    }

    static List<Mat> LoadImages(List<string> ImageList)
    {
        List<Mat> images = new List<Mat>();

        foreach (var imgFile in ImageList)
        {
            Mat img = Cv2.ImRead(imgFile);  // Load image
            if (img.Empty())
            {
                Console.WriteLine($"Error loading image: {imgFile}");
            }
            else
            {
                images.Add(img); 
                Console.WriteLine($"Loading image: {imgFile}");
            }
        }

        return images;
    }

    static float[] PreprocessImages(List<Mat> Ori_images, Size pad_size, bool to_rgb = true)
    {
        Console.WriteLine("Preprocessing images...");
        int rgb_img_length = 3 * pad_size.Height * pad_size.Width;
        float[] preprocessedImages = new float[Ori_images.Count* rgb_img_length];
        Mat processed_img;

        for (int i = 0; i < Ori_images.Count; i++)
        {
            processed_img  = Ori_images[i].Clone();
            if (to_rgb)
            {
                Cv2.CvtColor(processed_img, processed_img, ColorConversionCodes.BGR2RGB); //because od dinov2 backbone is use RGB channel image training
            }

            processed_img = Padding(processed_img, pad_size); // Pad image

            // Preprocess the image (normalize and convert to float array/tensor data)
            var tensorData = PreprocessImage(processed_img);
            Array.Copy(tensorData, 0, preprocessedImages, i * rgb_img_length, rgb_img_length);
        }

        return preprocessedImages; 
    }

    static float[] mean = { 0.485f, 0.456f, 0.406f };
    static float[] std = { 0.229f, 0.224f, 0.225f };
    static float[] PreprocessImage(Mat img)
    {
        float[] data = new float[3 * img.Rows * img.Cols];

        // Normalize each pixel and populate the array
        for (int y = 0; y < img.Rows; y++)
        {
            for (int x = 0; x < img.Cols; x++)
            {
                Vec3b pixel = img.At<Vec3b>(y, x); // Get the pixel at (y, x)

                // Normalize R, G, B values
                data[y * img.Cols + x] = (pixel.Item2 / 255.0f - mean[0]) / std[0];                             // R channel
                data[img.Cols * img.Rows + y * img.Cols + x] = (pixel.Item1 / 255.0f - mean[1]) / std[1];       // G channel
                data[2 * img.Cols * img.Rows + y * img.Cols + x] = (pixel.Item0 / 255.0f - mean[2]) / std[2];   // B channel
            }
        }

        return data; 
    }

    static Mat ConvertToColoredImage(Mat logits)
    {
        // Create a 3-channel Mat (initially all zeros)
        Mat colorImage = new Mat(logits.Size(), MatType.CV_8UC3);
        Vec3b[] colormapArray = new Vec3b[] { new Vec3b(50, 50, 50), new Vec3b(0, 0, 255) };

        // Iterate through the binary mask
        for (int y = 0; y < logits.Rows; y++)
        {
            for (int x = 0; x < logits.Cols; x++)
            {
                // Get the value of the pixel in the binary mask (0 or 1)
                byte value = logits.At<byte>(y, x);

                // Map the value to a 3-channel color and set it to the output image
                colorImage.Set(y, x, colormapArray[value]);
            }
        }

        return colorImage;
    }

    static void visualize(List<Mat> logits, List<Mat> ori_images, List<string> image_list, List<string> predicts, List<List<Rect>> bboxes)
    {
        
        double alpha = 0.5;
        for (int i = 0; i < ori_images.Count; i++)
        {
            for (int j = 0; j < bboxes[i].Count; j++)
            {
                Cv2.Rectangle(ori_images[i], bboxes[i][j].BottomRight, bboxes[i][j].TopLeft, new Scalar(0, 255, 0), 2);
            }


            var visualizeSegmentation = ConvertToColoredImage(logits[i]);

            Cv2.AddWeighted(ori_images[i], alpha, visualizeSegmentation, (1-alpha), 0, visualizeSegmentation);
            Cv2.ImShow(image_list[i], visualizeSegmentation);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }
        
    }
    

    static void Main()
    {
        // =========================================================================================== //
        // Parameters (initialize inference session)
        string HEAD_TYPE = "ms";    // only provide ms model
        bool is_fp16 = true;        // for onnx model is_fp16
        int now_device = 0;         // GPU
        
        // if empty string, onnx_model_path will automatically generate by HEAD_TYPE and is_fp16
        // for example: onnx_model_path = @"{model_path}\dinov2b14_ms_1_fp16.onnx"
        // string onnx_model_path = @"D:\AUOCrackCSAPI\AUOCrackCSAPI\onnx_model\dinov2b14_ms_1_fp16.onnx"; 
        string onnx_model_path = @"D:\AUOCrackCSAPI\AUOCrackCSAPI\onnx_model\dinov2b14_ms_1_fp16.onnx";

        // Parameters (Load data for inference)
        string image_path = @"D:\AUOCrackCSAPI\AUOCrackCSAPI\img_644\"; // input image path can be a directory or a single image
        
        // Parameters (inference)
        int batch_size = 8;             // inference batch size

        // Parameters (output)
        int seg_area_threshold = 100;   // filter out segments with small area size smaller than seg_area_threshold
        bool is_dilation = true;        // dilate the segmentation mask
        bool return_logits = true;      // return segmentation mask or not
        bool show_result = true;       // show visualization result or not, only work with return_logits = true
        // =========================================================================================== //


        // Load images and preprocess (only for demo)
        List<string> image_list = GetImageList(image_path); 
        List<Mat> ori_images = LoadImages(image_list);
        float[] preprocessedImages = PreprocessImages(ori_images, new Size(644, 644));

        
        // Load onnx model
        var onnxInference = new OnnxInference(); 
        onnxInference.LoadModel(onnx_model_path, HEAD_TYPE, now_device, is_fp16, batch_size); 


        // Inference
        List<Mat> outputMats;
        if (is_fp16)
        {
            Float16[] preprocessedImagesFP16 = preprocessedImages.Select(y => (Float16)y).ToArray(); // fp32 to fp16 preprocessedImages
            outputMats = onnxInference.InferenceFP16(preprocessedImagesFP16, batch_size);
        }
        else
        {
            outputMats = onnxInference.Inference(preprocessedImages, batch_size);
        }

        

        // Output
        var result = onnxInference.Output(outputMats, ori_images, seg_area_threshold, is_fp16, is_dilation, return_logits);
        List<string> predicts = result.Item1; //["OK","OK","NG","NG",...]
        List<List<Rect>> bboxes = result.Item2; //[[[x1, y1, x2, y2],[x1, y1, x2, y2],...], [[x1, y1, x2, y2],[x1, y1, x2, y2],...]

        if (return_logits)
        {
            List<Mat> logits = result.Item3; // segmentation result with logits(0:bg ,1:crack)

            // Visualize (only for demo)
            if (show_result)
            {
                visualize(outputMats, ori_images, image_list, predicts, bboxes);
            }
        }
        
    }
}