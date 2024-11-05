package uk.ac.soton.ecs.vision;

import java.util.ArrayList;
import java.util.Arrays;

public class processingData {
    float[] stdv;
    float[] mean;
    ArrayList<vTuple> data;

    // tuple allowing me to return the mean image, standard deviation of the images and
    public processingData(ArrayList<vTuple> data, float[] mean, float[]stdv) {
        this.data = data;
        this.mean = mean;
        this.stdv = stdv;
    }

    //return the standard deviation
    public float[] getStdv(){
        return stdv;
    }

    //return the mean image
    public float[] getMean(){
        return mean;
    }

    //return the manipulated data that has the 0 mean and unit length
    public ArrayList<vTuple> getData(){
        return data;
    }
}