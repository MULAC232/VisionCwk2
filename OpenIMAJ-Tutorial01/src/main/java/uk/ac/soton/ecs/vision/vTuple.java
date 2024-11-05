package uk.ac.soton.ecs.vision;
//this class holds a tuple object with an image's classification and pixel vector
public class vTuple {
    float[] vector;
    String type;

    public vTuple(float[] vector, String type) {
        this.type = type;
        this.vector = vector;
    }

    //return the tuple's vector
    public float[] getVector() {
        return vector;
    }

    //return the tuple's classification
    public String getType() {
        return type;
    }

    //set a new vector value
    public void setVector(float[] a){
        vector = a;
    }
}
