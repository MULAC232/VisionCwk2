package uk.ac.soton.ecs.vision;

import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.MBFImage;
import org.openimaj.image.colour.ColourSpace;
import org.openimaj.image.colour.RGBColour;
import org.openimaj.image.processing.convolution.FGaussianConvolve;
import org.openimaj.image.typography.hershey.HersheyFont;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.MemoryImageSource;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.Buffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.Collectors;

/**
 * OpenIMAJ Hello world!
 *
 */
public class KNearestNeighbour_Run1 {

    public static void main(String[] args) throws IOException {
        //defines the size that the images are resized to
        int GRIDSIZE = 4;

        //file path for the training data
        File folder = new File("OpenIMAJ-Tutorial01" + File.separator + "src" + File.separator + "training" + File.separator + "training");
        File[] types = folder.listFiles(f -> !f.isHidden());
        int length = types.length;
        ArrayList<vTuple> data = new ArrayList<vTuple>();

        //retrieves the training data from the file
        for(int i=0; i<length; i++){
            try {
                File subFolder = types[i];
                File[] training = subFolder.listFiles(f -> !f.isHidden());
                String type = subFolder.getName();
                int length2 = training.length;
                for (int n = 0; n < length2; n++) {
                    File file = training[n];
                    float[] vector = getImage(file, GRIDSIZE);
                    vTuple tuple = new vTuple(vector, type);
                    data.add(tuple);
                }
            }catch(Exception e){
                System.out.println("error reading directory");
            }
        }
        //applies 0 mean and unit length to the vectors
        processingData pd = process(data, GRIDSIZE);
        ArrayList<vTuple> data2 = pd.getData();

        //retrieves the test data from the file
        File testFolder = new File("OpenIMAJ-Tutorial01" + File.separator + "src" + File.separator + "testing" + File.separator + "testing");
        File[] testFiles = testFolder.listFiles(f -> !f.isHidden());
        // Sort files numerically
        Arrays.sort(testFiles, new Comparator<File>() {
            @Override
            public int compare(File o1, File o2) {
                return getNumber(o1.getName()) - getNumber(o2.getName());
            }

            private int getNumber(String fileName) {
                String numString = fileName.split("\\.")[0];
                int number = Integer.parseInt(numString);
                return number;
            }
        });

        //the try block takes the test image, processes it and runs it against the training data to predict its class
        try {
            FileWriter writer = new FileWriter("run1.txt");
            for (File current : testFiles) {
                float[] test = getImage(current, GRIDSIZE);
                float[] test2 = sub(test, pd.getMean());
                float[] test3 = div(test2, pd.getStdv());
                distanceTuple[] distances = calc(data2, test3);
                String decision = sort(distances);
                writer.write(current.getName() + " " + decision + "\n");
            }
            writer.close();
        } catch (IOException e) {
            System.out.println("error writing to file");
        }
    }

    //calculates the mean image vector
    public static float[] calcMean(ArrayList<vTuple> vectors, int GRIDSIZE){
        int length = vectors.size();
        float[] mean = new float[GRIDSIZE * GRIDSIZE];
        java.util.Arrays.fill(mean, 0);
        for(int i=0; i < length ; i++){
            for(int n=0; n<(GRIDSIZE*GRIDSIZE); n++){
                mean[n] = mean[n] + vectors.get(i).getVector()[n];
            }
        }
        for(int i=0; i<(GRIDSIZE*GRIDSIZE); i++){
            mean[i] = mean[i] / length;
        }
        return mean;
    }

    //allows you to add 2 vectors together
    public static float[] vAdd(float[] v1, float[] v2){
        int length = v1.length;
        float[] total = new float[length];
        for(int i=0; i<length; i++){
            total[i] = v1[i] + v2[i];
        }
        return total;
    }

    //allows you to subtract one vector from another
    public static float[] sub(float[] v1, float[] v2){
        int length = v1.length;
        float[] total = new float[length];
        for(int i=0; i<length; i++){
            total[i] = v1[i] - v2[i];
        }
        return total;
    }

    //allows you to divide one vector by another
    public static float[] div(float[] v1, float[] v2){
        int length = v1.length;
        float[] total = new float[length];
        for(int i=0; i<length; i++){
            total[i] = v1[i] / v2[i];
        }
        return total;
    }

    //allows you to divide a vector by a single value
    public static float[] div2(float[] v1, float num){
        int length = v1.length;
        float[] total = new float[length];
        for(int i=0; i<length; i++){
            total[i] = v1[i] / num;
        }
        return total;
    }

    //allows you to raise a vector to a power
    public static float[] pow(float[] v1){
        int length = v1.length;
        float[] total = new float[length];
        for(int i=0; i<length; i++){
            total[i] = v1[i] * v1[i];
        }
        return total;
    }

    //finds the square root of a vector
    public static float[] root(float[] v1){
        int length = v1.length;
        float[] total = new float[length];
        for(int i=0; i<length; i++){
            total[i] = (float) Math.sqrt(v1[i]);
        }
        return total;
    }

    //calculates the mean image and the image set's standard deviation
    //calculates the new image vectors, applying 0 mean and unit length
    public static processingData process(ArrayList<vTuple> data, int GRIDSIZE){
        float[] mean = calcMean(data, GRIDSIZE);
        float[] total = new float[GRIDSIZE*GRIDSIZE];
        java.util.Arrays.fill(mean, 0);
        int length = data.size();
        for(int i=0; i<length; i++){
            float[] v = data.get(i).getVector();
            float[] v2 = sub(v, mean);
            data.get(i).setVector(v2);
            float[] square = pow(v2);
            total = vAdd(total, square);
        }

        float[] stdv = root(div2(total, length));

        for(int i=0; i<length; i++){
            float[] v = data.get(i).getVector();
            float[] v2 = div(v, stdv);
            data.get(i).setVector(v2);
        }
        processingData pd = new processingData(data, mean, stdv);

        return pd;
    }

    //pulls the image file from memory
    //cuts the image to a square around the centre
    //uses the resize function to resize the image to a small grid
    public static float[] getImage(File file, int GRIDSIZE){
        FImage image = null;
        int width = 0;
        int height = 0;
        float[] vector = new float[GRIDSIZE*GRIDSIZE];
        try {
            image = ImageUtilities.readF(file);
            width = image.getWidth();
            height = image.getHeight();
            FImage newImage;
            if(width < height){
                newImage = image.extractCenter(width, width);
            }
            else{
                newImage = image.extractCenter(height, height);
            }
            FImage finalImage = null;
            finalImage = resize(newImage, GRIDSIZE);
            finalImage.getPixelVectorNative(vector);
        } catch (IOException e) {
            System.out.println("error reading file");
        }
        return vector;
    }

    //resizes the image to a grid of (size x size) pixels
    public static FImage resize(FImage image, int size) throws IOException {
        BufferedImage image2 = ImageUtilities.createBufferedImage(image);
        BufferedImage background = new BufferedImage(size, size, BufferedImage.TRANSLUCENT);
        Graphics2D oldImage = background.createGraphics();
        oldImage.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        oldImage.drawImage(image2, 0,0,size, size, null);
        oldImage.dispose();
        FImage finalImage = ImageUtilities.createFImage(background);
        return finalImage;
    }

    //work out euclidean distance between 2 vectors
    public static float euclidean(vTuple v1, float[] v2){
        int length = v1.getVector().length;
        float total = 0;
        for(int i=0; i<length; i++){
            float difference = v1.getVector()[i] - v2[i];
            float square = difference * difference;
            total = total + square;
        }
        return (float) Math.sqrt(total);
    }

    //finds the difference between the training data and test image using the euclidean function
    public static distanceTuple[] calc(ArrayList<vTuple> vectors, float[] test){
        int length = vectors.size();
        distanceTuple[] distances = new distanceTuple[length];
        for(int i=0; i<length; i++){
            distances[i] = new distanceTuple(euclidean(vectors.get(i), test), vectors.get(i).getType());

        }
        return distances;
    }

    //pulls the closest k images from the list
    //uses the mode function to find the most popular classification
    public static String vote(int k, distanceTuple[] vectors){
        String[] closest = new String[k];
        for(int i=0; i<k; i++){
            closest[i] = vectors[i].getType();
        }
        return mode(closest);
    }

    //sorts the training images by distance from the test image (lowest to highest)
    public static String sort(distanceTuple[] distances){
        distanceTuple[] sorted = Arrays.stream(distances).sorted().collect(Collectors.toList()).toArray(new distanceTuple[0]);
        return vote(15, sorted);
    }

    //finds the most popular class within the selected images from the training set and returns it
    //this is the classification given to the image we are predicting
    public static String mode(String[] types){
        int length = types.length;
        ArrayList<String> type = new ArrayList();
        int count = 0;
        int currentCount;
        String currentType = "";
        for(int i=0; i<length; i++){
            currentType = types[i];
            currentCount = 0;
            for(int n=0; n<length; n++){
                if(types[n].equals(currentType)){
                    currentCount ++;
                }
            }
            if(currentCount > count){
                count = currentCount;
                type.add(0, currentType);
            }
        }
        return type.get(0);
    }


}

