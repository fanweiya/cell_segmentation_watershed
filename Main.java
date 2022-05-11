//import org.opencv.core.*;
//import org.opencv.imgcodecs.Imgcodecs;
//import org.opencv.highgui.HighGui;
//import org.opencv.imgproc.Imgproc;
//
//import java.util.ArrayList;
//import java.util.List;
//import java.util.Random;
//
//public class Main {
//    static {
//        //加载库
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//    }
//    public static Mat watershed_process(Mat imgResult){
//        // Create binary image from source image
//        Imgproc.cvtColor(imgResult, imgResult, Imgproc.COLOR_BGRA2BGR);
//        Mat bw = new Mat();
//        Imgproc.cvtColor(imgResult, bw, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.threshold(bw, bw, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
//        //Imgcodecs.imwrite(dirPath+"/"+"bw.jpg",bw);
//        // Perform the distance transform algorithm
//        Mat dist = new Mat();
//        Imgproc.distanceTransform(bw, dist, Imgproc.DIST_L2, 3);
//        imwrite("distanceTransform",dist);
//        // Normalize the distance image for range = {0.0, 1.0}
//        // so we can visualize and threshold it
//        Core.normalize(dist, dist, 0.0, 1.0, Core.NORM_MINMAX);
//        imwrite("normalize", dist);
//        Mat distDisplayScaled = new Mat();
//        Core.multiply(dist, new Scalar(255), distDisplayScaled);
//        Mat distDisplay = new Mat();
//        distDisplayScaled.convertTo(distDisplay, CvType.CV_8U);
//        // Threshold to obtain the peaks
//        // This will be the markers for the foreground objects
//        Imgproc.threshold(dist, dist, 0, 1.0, Imgproc.THRESH_BINARY);
//        imwrite("threshold", dist);
//        // Dilate a bit the dist image
//        Mat kernel1 = Mat.ones(5, 5, CvType.CV_8U);
//        Imgproc.erode(dist, dist, kernel1);
//        Mat distDisplay2 = new Mat();
//        dist.convertTo(distDisplay2, CvType.CV_8U);
//        Core.multiply(distDisplay2, new Scalar(255), distDisplay2);
//        // Create the CV_8U version of the distance image
//        // It is needed for findContours()
//        Mat dist_8u = new Mat();
//        dist.convertTo(dist_8u, CvType.CV_8U);
//        // Find total markers
//        List<MatOfPoint> contours = new ArrayList<>();
//        Mat hierarchy = new Mat();
//        Imgproc.findContours(dist_8u, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
//        // Create the marker image for the watershed algorithm
//        Mat markers = Mat.zeros(dist.size(), CvType.CV_32S);
//        // Draw the foreground markers
//        for (int i = 0; i < contours.size(); i++) {
//            Imgproc.drawContours(markers, contours, i, new Scalar(i + 1), -1);
//        }
//        // Draw the background marker
//        Mat markersScaled = new Mat();
//        markers.convertTo(markersScaled, CvType.CV_32F);
//        Core.normalize(markersScaled, markersScaled, 0.0, 255.0, Core.NORM_MINMAX);
//        Imgproc.circle(markersScaled, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
//        Mat markersDisplay = new Mat();
//        markersScaled.convertTo(markersDisplay, CvType.CV_8U);
//        Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
//
//        // Perform the watershed algorithm
//        Imgproc.watershed(imgResult, markers);
//        // Generate random colors
////        Random rng = new Random(12345);
////        List<Scalar> colors = new ArrayList<>(contours.size());
////        for (int i = 0; i < contours.size(); i++) {
////            int b = rng.nextInt(256);
////            int g = rng.nextInt(256);
////            int r = rng.nextInt(256);
////            colors.add(new Scalar(b, g, r));
////        }
//        List<MatOfPoint> end_contours = new ArrayList<>();
//        for(int obj = 1; obj < contours.size() + 1; obj++)
//        {
//            Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC1);
//            byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
//            dst.get(0, 0, dstData);
//            // Fill labeled objects with random colors
//            int[] markersData = new int[(int) (markers.total() * markers.channels())];
//            markers.get(0, 0, markersData);
//            for (int i = 0; i < markers.rows(); i++) {
//                for (int j = 0; j < markers.cols(); j++) {
//                    int index = markersData[i * markers.cols() + j];
//                    if (index== obj && index <= contours.size()) {
//                        dstData[i * dst.cols() + j] = (byte) 255;
//                    } else {
//                        dstData[i * dst.cols() + j] = 0;
//                    }
//                }
//            }
//            dst.put(0, 0, dstData);
//            //Mat graymat=new Mat();
//            List<MatOfPoint> temp_contours = new ArrayList<>();
//            //Imgproc.cvtColor(dst, graymat, Imgproc.COLOR_BGR2GRAY);
//            Imgproc.findContours(dst,temp_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
//            MatOfPoint temp_tours=temp_contours.get(0);
//            end_contours.add(temp_tours);
//        }
//        Mat mark = Mat.zeros(markers.size(), CvType.CV_8U);
//        markers.convertTo(mark, CvType.CV_8UC1);
//        Core.bitwise_not(mark, mark);
//        // imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
//        // image looks like at that point
//        // Generate random colors
//        Random rng = new Random(12345);
//        List<Scalar> colors = new ArrayList<>(contours.size());
//        for (int i = 0; i < contours.size(); i++) {
//            int b = rng.nextInt(256);
//            int g = rng.nextInt(256);
//            int r = rng.nextInt(256);
//            colors.add(new Scalar(b, g, r));
//        }
//        // Create the result image
//        Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC3);
//        byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
//        dst.get(0, 0, dstData);
//        // Fill labeled objects with random colors
//        int[] markersData = new int[(int) (markers.total() * markers.channels())];
//        markers.get(0, 0, markersData);
//        for (int i = 0; i < markers.rows(); i++) {
//            for (int j = 0; j < markers.cols(); j++) {
//                int index = markersData[i * markers.cols() + j];
//                if (index > 0 && index <= contours.size()) {
//                    dstData[(i * dst.cols() + j) * 3 + 0] = (byte) colors.get(index - 1).val[0];
//                    dstData[(i * dst.cols() + j) * 3 + 1] = (byte) colors.get(index - 1).val[1];
//                    dstData[(i * dst.cols() + j) * 3 + 2] = (byte) colors.get(index - 1).val[2];
//                } else {
//                    dstData[(i * dst.cols() + j) * 3 + 0] = 0;
//                    dstData[(i * dst.cols() + j) * 3 + 1] = 0;
//                    dstData[(i * dst.cols() + j) * 3 + 2] = 0;
//                }
//            }
//        }
//        dst.put(0, 0, dstData);
//        return dst;
//        //return end_contours;
//    }
//    public static void imwrite(String name, Mat image) {
//        Mat show = image.clone();
//        Imgcodecs.imwrite(name+".jpg", show);
//    }
//    public static void main(String[] args) {
//        //读取图像
////        Mat image = Imgcodecs.imread("src/1.jpg");
////        Mat srcmask = Imgcodecs.imread("src/maskImagemat.jpg");
//        Mat image = Imgcodecs.imread("src/1634003149613.jpg");
//        Mat imgResult = Imgcodecs.imread("src/maskImagemat2.jpg");
//
//        Mat result=watershed_process(imgResult);
//        //显示图像
//        Core.addWeighted(image, 0.5, result, 0.5, 0.0, result);
//        //设置显示窗口大小
//        Imgproc.resize(result, result, new Size(result.cols() * 0.25, result.rows() * 0.25));
//        HighGui.imshow("result",result);
//        HighGui.waitKey(0);
//    }
//
//}
import java.util.*;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;


class ImageSegmentation {
    public List<MatOfPoint> flitercontours(List<MatOfPoint> A, List<MatOfPoint> B){
        //判断B轮廓的点是否在A的轮廓中，如果在，则删除A中的轮廓，返回A+B
        //System.out.println("A.size()="+A.size()+" B.size()="+B.size());
        for(int i=0;i<B.size();i++){
            MatOfPoint b=B.get(i);
            for(int j=0;j<A.size();j++){
                MatOfPoint a=A.get(j);
                MatOfPoint2f a2f=new MatOfPoint2f(a.toArray());
                if(Imgproc.pointPolygonTest(a2f,b.toArray()[0],false)>=0){
                    A.remove(j);
                }
            }
        }
        //System.out.println("A.size()="+A.size()+" B.size()="+B.size());
        List<MatOfPoint> C=new ArrayList<MatOfPoint>();
        C.addAll(A);
        C.addAll(B);
        return C;
    }
    public List<MatOfPoint> fastGetCouter(Mat markers,int CouterNUM){
        List<MatOfPoint> end_contours = new ArrayList<>();
        Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC1);
        byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
        dst.get(0, 0, dstData);
        // Fill labeled objects with random colors
        int[] markersData1 = new int[(int) (markers.total() * markers.channels())];
        markers.get(0, 0, markersData1);
        for (int i = 0; i < markers.rows(); i++) {
            for (int j = 0; j < markers.cols(); j++) {
                int index = markersData1[i * markers.cols() + j];
                if (index > 0 && index <= CouterNUM) {
                    dstData[i * dst.cols() + j] = (byte) index;
                } else {
                    dstData[i * dst.cols() + j] = 0;
                }
            }
        }
        dst.put(0, 0, dstData);
        for(int obj = 1; obj < CouterNUM + 1; obj++) {
            Mat objMask=new Mat();
            Core.inRange(dst, new Scalar(obj), new Scalar(obj), objMask);
            List<MatOfPoint> temp_contours = new ArrayList<>();
            Imgproc.findContours(objMask,temp_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
            if(temp_contours.size()>0){
                MatOfPoint temp_tours=temp_contours.get(0);
                end_contours.add(temp_tours);
            }
        }
        return end_contours;
    }
    public List<MatOfPoint> fastGetCouter2(Mat markers,int CouterNUM){ //796
        List<MatOfPoint> end_contours = new ArrayList<>();
        List<List<Integer>> partsLists = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        for(int obj = 1; obj < CouterNUM + 1; obj++)
        {
            temp.add(obj);
            if(obj%255==0){
                partsLists.add(temp);
                temp=new ArrayList<>();
            }
        }
        partsLists.add(temp);
//        int testzero = 0;
//        for(List<Integer> l:partsLists){
//            String temp_str="";
//            String temp_str1="";
//            for(int i:l){
//                temp_str=temp_str+ i +",";
//                temp_str1=temp_str1+ (i-255*testzero) +" ";
//            }
//            testzero++;
//            System.out.println(temp_str+"\n"+temp_str1);
//        }
        for(int z=0;z<partsLists.size();z++){
            List<Integer> partlist=partsLists.get(z);
            Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC1);
            byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
            dst.get(0, 0, dstData);
            // Fill labeled objects with random colors
            int[] markersData = new int[(int) (markers.total() * markers.channels())];
            markers.get(0, 0, markersData);
            for (int i = 0; i < markers.rows(); i++) {
                for (int j = 0; j < markers.cols(); j++) {
                    int index = markersData[i * markers.cols() + j];
                    //System.out.println(index+" "+partlist.contains(index)+" z="+z+" index-255*z="+(index-255*z));
                    if (partlist.contains(index)) {
                        dstData[i * dst.cols() + j] = (byte) (index-255*z);
                    } else {
                        dstData[i * dst.cols() + j] = 0;
                    }
                }
            }
            dst.put(0, 0, dstData);
            for(int objj = 1; objj < 256; objj++) {
                Mat objMask=new Mat();
                Core.inRange(dst, new Scalar(objj), new Scalar(objj), objMask);
                List<MatOfPoint> temp_contours = new ArrayList<>();
                Imgproc.findContours(objMask,temp_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
                if(temp_contours.size()>0){
                    MatOfPoint temp_tours=temp_contours.get(0);
                    end_contours.add(temp_tours);
                }
            }
        }
        return end_contours;
    }
    public List<MatOfPoint> fastGetCouter3(Mat markers,int CouterNUM){
        List<MatOfPoint> end_contours = new ArrayList<>();
        for(int obj = 1; obj < CouterNUM + 1; obj++)
        {
            Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC1);
            byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
            dst.get(0, 0, dstData);
            // Fill labeled objects with random colors
            int[] markersData = new int[(int) (markers.total() * markers.channels())];
            markers.get(0, 0, markersData);
            for (int i = 0; i < markers.rows(); i++) {
                for (int j = 0; j < markers.cols(); j++) {
                    int index = markersData[i * markers.cols() + j];
                    if (index== obj && index <= CouterNUM) {
                        dstData[i * dst.cols() + j] = (byte) 255;
                    } else {
                        dstData[i * dst.cols() + j] = 0;
                    }
                }
            }
            dst.put(0, 0, dstData);
            //Mat graymat=new Mat();
            List<MatOfPoint> temp_contours = new ArrayList<>();
            //Imgproc.cvtColor(dst, graymat, Imgproc.COLOR_BGR2GRAY);
            Imgproc.findContours(dst,temp_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
            if(temp_contours.size()>0){
                MatOfPoint temp_tours=temp_contours.get(0);
                end_contours.add(temp_tours);
            }
        }
        return end_contours;
    }
    public void run(String[] args) {
        // Load the image
        Mat srcimage=Imgcodecs.imread("src/1_hstack.jpg");
        Mat srcOriginal = Imgcodecs.imread("src/maskImagemat_hstack.jpg");
        //Imgproc.resize(srcOriginal, srcOriginal, new Size(srcOriginal.cols() * 0.25, srcOriginal.rows() * 0.25));
        if (srcOriginal.empty()) {
            System.err.println("Cannot read image: ");
            System.exit(0);
        }
        // Show source image
        //HighGui.imshow("Source Image", srcOriginal);
        // Change the background from white to black, since that will help later to
        // extract
        // better results during the use of Distance Transform
        Mat bw = new Mat();
        Imgproc.cvtColor(srcOriginal, bw, Imgproc.COLOR_BGRA2GRAY);
        Imgproc.threshold(bw, bw, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        //HighGui.imshow("Binary Image", bw);
        List<MatOfPoint> temp_contours1 = new ArrayList<>();
        Imgproc.findContours(bw,temp_contours1,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
        // Perform the distance transform algorithm
        Mat dist = new Mat();
        Imgproc.distanceTransform(bw, dist, Imgproc.DIST_L2, 3);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        Core.normalize(dist, dist, 0.000, 1.000, Core.NORM_MINMAX);
        Mat distDisplayScaled = new Mat();
        Core.multiply(dist, new Scalar(255), distDisplayScaled);
        Mat distDisplay = new Mat();
        distDisplayScaled.convertTo(distDisplay, CvType.CV_8U);
        //HighGui.imshow("Distance Transform Image", distDisplay);
        // Threshold to obtain the peaks
        // This will be the markers for the foreground objects
        Imgproc.threshold(dist, dist, 0.5, 1.0, Imgproc.THRESH_BINARY);
        // Dilate a bit the dist image
        Mat kernel1 = Mat.ones(7, 7, CvType.CV_8U);
        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(7, 7));
//        Imgproc.dilate(dist, dist, kernel1);
        //Imgproc.erode(dist,dist,kernel1,new Point(-1,-1),4);
        Imgproc.morphologyEx(dist, dist, Imgproc.MORPH_CLOSE, kernel2, new Point(-1, -1), 1);
        Mat distDisplay2 = new Mat();
        dist.convertTo(distDisplay2, CvType.CV_8U);
        Core.multiply(distDisplay2, new Scalar(255), distDisplay2);
        //HighGui.imshow("Peaks", distDisplay2);
        // Create the CV_8U version of the distance image
        // It is needed for findContours()
        Mat dist_8u = new Mat();
        dist.convertTo(dist_8u, CvType.CV_8U);
        // Find total markers
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(dist_8u, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        // Create the marker image for the watershed algorithm
        Mat markers = Mat.zeros(dist.size(), CvType.CV_32S);
        // Draw the foreground markers
        contours=flitercontours(temp_contours1,contours);
        for (int i = 0; i < contours.size(); i++) {
            Imgproc.drawContours(markers, contours, i, new Scalar(i + 1), -1);
        }
        // Draw the background marker
        Mat markersScaled = new Mat();
        markers.convertTo(markersScaled, CvType.CV_32F);
        Core.normalize(markersScaled, markersScaled, 0.0, 255.0, Core.NORM_MINMAX);
        Imgproc.circle(markersScaled, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
        Mat markersDisplay = new Mat();
        markersScaled.convertTo(markersDisplay, CvType.CV_8U);
        //HighGui.imshow("Markers", markersDisplay);
        Imgproc.circle(markers, new Point(5, 5), 3, new Scalar(255, 255, 255), -1);
        // Perform the watershed algorithm

        Imgproc.watershed(srcOriginal, markers);
        //Generate the final image
//        //get each contour
//        List<MatOfPoint> end_contours = new ArrayList<>();
//        for(int obj = 1; obj < contours.size() + 1; obj++)
//        {
//            Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC1);
//            byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
//            dst.get(0, 0, dstData);
//            // Fill labeled objects with random colors
//            int[] markersData = new int[(int) (markers.total() * markers.channels())];
//            markers.get(0, 0, markersData);
//            for (int i = 0; i < markers.rows(); i++) {
//                for (int j = 0; j < markers.cols(); j++) {
//                    int index = markersData[i * markers.cols() + j];
//                    if (index== obj && index <= contours.size()) {
//                        dstData[i * dst.cols() + j] = (byte) 255;
//                    } else {
//                        dstData[i * dst.cols() + j] = 0;
//                    }
//                }
//            }
//            dst.put(0, 0, dstData);
//            //Mat graymat=new Mat();
//            List<MatOfPoint> temp_contours = new ArrayList<>();
//            //Imgproc.cvtColor(dst, graymat, Imgproc.COLOR_BGR2GRAY);
//            Imgproc.findContours(dst,temp_contours,new Mat(),Imgproc.RETR_EXTERNAL,Imgproc.CHAIN_APPROX_SIMPLE);
//            MatOfPoint temp_tours=temp_contours.get(0);
//            end_contours.add(temp_tours);
//        }
//        List<MatOfPoint> end_contours = new ArrayList<>();
//        end_contours=fastGetCouter(markers,contours.size());
//        System.out.println("end_contours.size()="+end_contours.size());
//        List<MatOfPoint> end_contours2 = new ArrayList<>();
        Date date = new Date();
        contours=fastGetCouter2(markers,contours.size());
        System.out.println("end_contours2.size()="+contours.size());
        Date date2 = new Date();
        System.out.println("fastGetCouter2 time="+(date2.getTime()-date.getTime()));
//        List<MatOfPoint> end_contours3 = new ArrayList<>();
//        end_contours3=fastGetCouter3(markers,contours.size());
//        System.out.println("end_contours3.size()="+end_contours3.size());
//        for(int i=0;i<end_contours2.size();i++){
//            Imgproc.drawContours(srcimage,end_contours2,i,new Scalar(255,0,0),2);
//            Imgproc.putText(srcimage,String.valueOf(i),end_contours2.get(i).toList().get(0),Imgproc.FONT_HERSHEY_SIMPLEX,1,new Scalar(255,0,0),2);
//        }
//        for(int i=0;i<end_contours3.size();i++){
//            Imgproc.drawContours(srcimage,end_contours3,i,new Scalar(0,255,0),1);
//            Imgproc.putText(srcimage,String.valueOf(i),end_contours3.get(i).toList().get(0),Imgproc.FONT_HERSHEY_SIMPLEX,1,new Scalar(0,255,0),1);
//        }
//        System.out.println(temp_contours1.size()+" "+end_contours2.size()+" "+end_contours3.size());
        //contours=end_contours2;
////        Mat mark = Mat.zeros(markers.size(), CvType.CV_8U);
////        markers.convertTo(mark, CvType.CV_8UC1);
////        Core.bitwise_not(mark, mark);
////        HighGui.imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
//        // image looks like at that point
//        // Generate random colors

        Random rng = new Random(12345);
        List<Scalar> colors = new ArrayList<>(contours.size());
        for (int i = 0; i < contours.size(); i++) {
            int b = rng.nextInt(256);
            int g = rng.nextInt(256);
            int r = rng.nextInt(256);
            colors.add(new Scalar(b, g, r));
        }
        // Create the result image
        Mat dst = Mat.zeros(markers.size(), CvType.CV_8UC3);
        byte[] dstData = new byte[(int) (dst.total() * dst.channels())];
        dst.get(0, 0, dstData);
        // Fill labeled objects with random colors
        int[] markersData = new int[(int) (markers.total() * markers.channels())];
        markers.get(0, 0, markersData);
        for (int i = 0; i < markers.rows(); i++) {
            for (int j = 0; j < markers.cols(); j++) {
                int index = markersData[i * markers.cols() + j];
                if (index > 0 && index <= contours.size()) {
                    dstData[(i * dst.cols() + j) * 3 + 0] = (byte) colors.get(index - 1).val[0];
                    dstData[(i * dst.cols() + j) * 3 + 1] = (byte) colors.get(index - 1).val[1];
                    dstData[(i * dst.cols() + j) * 3 + 2] = (byte) colors.get(index - 1).val[2];
                } else {
                    dstData[(i * dst.cols() + j) * 3 + 0] = 0;
                    dstData[(i * dst.cols() + j) * 3 + 1] = 0;
                    dstData[(i * dst.cols() + j) * 3 + 2] = 0;
                }
            }
        }
        dst.put(0, 0, dstData);

        Core.addWeighted(srcimage, 0.5, dst, 0.5, 0.0, srcimage);
        Imgcodecs.imwrite("src/watershed.jpg", srcimage);
//        Imgproc.resize(dst,dst, new Size(dst.width()*0.75, dst.height()*0.75));
        //HighGui.imshow("Final Result", dst);
        //HighGui.imshow("Final Result", srcimage);
        //HighGui.waitKey();
        System.exit(0);
    }
}
public class Main {
    public static void main(String[] args) {
        // Load the native OpenCV library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        new ImageSegmentation().run(args);
    }
}