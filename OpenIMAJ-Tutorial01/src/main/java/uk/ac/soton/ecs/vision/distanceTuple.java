package uk.ac.soton.ecs.vision;

public class distanceTuple implements Comparable<distanceTuple> {
    float distance;
    String type;


    //stores an image's classification and the length from the vector to predict the class of
    public distanceTuple(float distance, String type) {
        this.type = type;
        this.distance = distance;
    }

    //return the distance from the vector
    public float getDistance() {
        return distance;
    }

    //return the image's classification
    public String getType() {
        return type;
    }

    //allows you to compare the tuples and thus sort a list of them
    @Override
    public int compareTo(distanceTuple o) {
        int num;
        float value = this.getDistance() - o.getDistance();
        if (value < 0) {
            num = -1;
        } else if (value > 0) {
            num = 1;
        } else {
            num = 0;
        }
        return num;
    }
}
