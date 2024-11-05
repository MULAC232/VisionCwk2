package uk.ac.soton.ecs.vision;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.GroupedDataset;
import org.openimaj.data.dataset.ListDataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.dataset.split.GroupedRandomSplitter;
import org.openimaj.experiment.evaluation.classification.ClassificationEvaluator;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMAnalyser;
import org.openimaj.experiment.evaluation.classification.analysers.confusionmatrix.CMResult;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;

/**
 * Class to handle the execution of Run2, including obtaining images, evaluating the accuracy of the
 * classifications and writing the results to run2.txt
 */
public class Run2 {

  public static void main(String[] args) {
    // Obtain training images
    GroupedDataset<String, VFSListDataset<FImage>, FImage> trainingImages = null;
    try {
      trainingImages = new VFSGroupDataset<>(
              new File(
                      "OpenIMAJ-Tutorial01"
                              + File.separator
                              + "src"
                              + File.separator
                              + "training"
                              + File.separator
                              + "training")
                      .getAbsolutePath(),
              ImageUtilities.FIMAGE_READER);
    } catch (FileSystemException e) {
      System.out.println("Error reading in training images");
    }
    // Split images into training, validation and testing
    GroupedRandomSplitter<String, FImage> trainValTestSplit = new GroupedRandomSplitter<>(
            trainingImages, 80, 0, 20);

    // Create classifier - adjust parameters to tune accuracy
    DenselySampledPixelPatchesClassifier denselySampledPixelPatchesClassifier = new DenselySampledPixelPatchesClassifier(
            trainValTestSplit.getTrainingDataset(), 8, 4, true, true, 1500, 300, 1.0, 0.00001);
    LiblinearAnnotator<FImage, String> annotator =
            denselySampledPixelPatchesClassifier.trainClassifier();

    // Test accuracy of classifier
    Run2.testClassifierAccuracy(trainValTestSplit.getTestDataset(), annotator);

    // Obtain testing images
    File testFolder = new File(
            "OpenIMAJ-Tutorial01" + File.separator + "src" + File.separator + "testing" + File.separator
                    + "testing");
    File[] testFiles = testFolder.listFiles(f -> !f.isHidden());
    // Write predictions of unseen images to text file
    Run2.writePredictions(annotator, testFiles, "run2.txt");
  }

  /**
   * Evaluate the accuracy of an annotator using the provided labelled test images
   *
   * @param testDataset Set of images to test the annotator on
   * @param annotator   Annotator to be evaluated
   */
  private static void testClassifierAccuracy(
          GroupedDataset<String, ListDataset<FImage>, FImage> testDataset,
          LiblinearAnnotator<FImage, String> annotator) {
    // Create evaluator
    ClassificationEvaluator<CMResult<String>, String, FImage> evaluator =
            new ClassificationEvaluator<>(
                    annotator,
                    testDataset,
                    new CMAnalyser<>(CMAnalyser.Strategy.SINGLE));
    // Evaluate accuracy
    Map<FImage, ClassificationResult<String>> guesses = evaluator.evaluate();
    CMResult<String> result = evaluator.analyse(guesses);
    System.out.println(result);
  }

  /**
   * Write predictions for a set of unlabelled images to a text file
   *
   * @param annotator Annotator to be used for classifying
   * @param testFiles Images to be classified
   * @param fileName  Name of file to write predictions to
   */
  private static void writePredictions(LiblinearAnnotator<FImage, String> annotator,
                                       File[] testFiles, String fileName) {
    // Sort files numerically
    assert testFiles != null;
    Arrays.sort(testFiles, new Comparator<>() {
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
    try {
      FileWriter writer = new FileWriter(fileName);
      for (File file : testFiles) {
        FImage testImage = ImageUtilities.readF(file);
        writer.write(file.getName() + " " + annotator.annotate(testImage).get(0).annotation + "\n");
      }
      writer.close();
    } catch (IOException e) {
      System.out.println("Error writing to file");
    }
  }

}
