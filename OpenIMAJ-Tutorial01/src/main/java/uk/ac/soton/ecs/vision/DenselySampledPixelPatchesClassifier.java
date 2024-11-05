package uk.ac.soton.ecs.vision;

import de.bwaldvogel.liblinear.SolverType;
import java.util.*;
import org.apache.commons.lang.ArrayUtils;
import org.openimaj.data.dataset.*;
import org.openimaj.experiment.dataset.sampling.GroupedUniformRandomisedSampler;
import org.openimaj.feature.*;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

/**
 * DenselySamplePixelPatchesClassifier class which handles the training of a linear classifier
 */
public class DenselySampledPixelPatchesClassifier {

  private final GroupedDataset<String, ListDataset<FImage>, FImage> trainingImages;
  private final int patchSize;
  private final int patchDifference;
  private final boolean normalisePatches;
  private final boolean meanCentrePatches;
  private final int numClusters;
  private final int clusteringSampleSize;
  private final double C;
  private final double epsilon;

  /**
   * DenselySamplePixelPatchesClassifier constructor
   *
   * @param trainingImages       Set of images used to train classifier
   * @param patchSize            Size of the patches extracted from images
   * @param patchDifference      Difference in pixels between consecutive patches
   * @param normalisePatches     Whether patches should be normalised
   * @param meanCentrePatches    Whether patches should be mean-centred
   * @param numClusters          The number of clusters used in the k-means clustering to obtain
   *                             visual words
   * @param clusteringSampleSize Number of images to use for the k-means clustering
   * @param C                    Penalty parameter for the classifier
   * @param epsilon              Tolerance of the classifier
   */
  public DenselySampledPixelPatchesClassifier(
          GroupedDataset<String, ListDataset<FImage>, FImage> trainingImages,
          int patchSize,
          int patchDifference,
          boolean normalisePatches,
          boolean meanCentrePatches,
          int numClusters,
          int clusteringSampleSize,
          double C,
          double epsilon) {
    this.trainingImages = trainingImages;
    this.patchSize = patchSize;
    this.patchDifference = patchDifference;
    this.normalisePatches = normalisePatches;
    this.meanCentrePatches = meanCentrePatches;
    this.numClusters = numClusters;
    this.clusteringSampleSize = clusteringSampleSize;
    this.C = C;
    this.epsilon = epsilon;
  }

  /**
   * Train an annotator using the instance variables
   *
   * @return Annotator that classifies images
   */
  public LiblinearAnnotator<FImage, String> trainClassifier() {
    // Learn vocabulary
    HardAssigner<float[], float[], IntFloatPair> hardAssigner = learnVocabulary();
    System.out.println("Learned vocabulary");

    // Train linear classifier
    FeatureExtractor<DoubleFV, FImage> featureExtractor =
            new DenselySampledPixelPatchesExtractor(
                    hardAssigner, patchSize, patchDifference, normalisePatches, meanCentrePatches);
    LiblinearAnnotator<FImage, String> annotator =
            new LiblinearAnnotator<>(
                    featureExtractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, C, epsilon);
    annotator.train(trainingImages);
    System.out.println("Created annotator");
    return annotator;
  }

  /**
   * Learn the visual words vocabulary
   *
   * @return Assigner which maps a patch to its closest visual word
   */
  private HardAssigner<float[], float[], IntFloatPair> learnVocabulary() {
    // Get patches from a random sample of the training images
    List<float[]> allPatches = new ArrayList<>();
    GroupedDataset<String, ListDataset<FImage>, FImage> sampleImages = GroupedUniformRandomisedSampler
            .sample(trainingImages, clusteringSampleSize);
    for (FImage image : sampleImages) {
      List<float[]> patches =
              DenselySampledPixelPatchesExtractor.getPatchesFromImage(
                      image, patchSize, patchDifference, normalisePatches, meanCentrePatches);
      allPatches.addAll(patches);
    }

    // Learn vocabulary and train vector quantisation
    return trainVectorQuantisation(allPatches);
  }

  /**
   * Train the vector quantisation used to assign a visual word to a patch
   *
   * @return Assigner which maps a provided patch to its closest visual word
   */
  private HardAssigner<float[], float[], IntFloatPair> trainVectorQuantisation(
          List<float[]> allPatches) {
    // Cluster patches to get visual words vocabulary
    FloatCentroidsResult clusters = clusterPatches(allPatches);
    // Create visual word assigner (assigns features to visual words)
    assert clusters != null;
    return clusters.defaultHardAssigner();
  }

  /**
   * K-means cluster the provided patches to create a vocabulary of the specified size
   *
   * @param patches Patches to cluster
   * @return Centroids resulting from the k-means clustering algorithm
   */
  private FloatCentroidsResult clusterPatches(List<float[]> patches) {
    FloatKMeans km = FloatKMeans.createKDTreeEnsemble(numClusters);
    return km.cluster(patches.toArray(new float[0][0]));
  }

  /**
   * DenselySampledPixelPatchesExtractor class that handles obtaining the densely-sampled pixel
   * patch BoVW feature from images
   */
  static class DenselySampledPixelPatchesExtractor implements FeatureExtractor<DoubleFV, FImage> {

    private final HardAssigner<float[], float[], IntFloatPair> hardAssigner;
    private final int patchSize;
    private final int patchDifference;
    private final boolean normalisePatches;
    private final boolean meanCentrePatches;

    /**
     * DenselySampledPixelPatchesExtractor constructor
     *
     * @param hardAssigner      Assigner which maps patches to visual words
     * @param patchSize         Size of the patches extracted from images
     * @param patchDifference   Sampling step between patches in pixels
     * @param normalisePatches  Whether patches should be normalised
     * @param meanCentrePatches Whether patches should be mean-centred
     */
    public DenselySampledPixelPatchesExtractor(
            HardAssigner<float[], float[], IntFloatPair> hardAssigner,
            int patchSize,
            int patchDifference,
            boolean normalisePatches,
            boolean meanCentrePatches) {
      this.hardAssigner = hardAssigner;
      this.patchSize = patchSize;
      this.patchDifference = patchDifference;
      this.normalisePatches = normalisePatches;
      this.meanCentrePatches = meanCentrePatches;
    }

    /**
     * Extract the DenselySampledPixelPatches feature from the given image
     *
     * @param imageObject Image to extract feature from
     * @return DenselySampledPixelPatches feature
     */
    public DoubleFV extractFeature(FImage imageObject) {
      List<float[]> patches =
              getPatchesFromImage(
                      imageObject, patchSize, patchDifference, normalisePatches, meanCentrePatches);
      BagOfVisualWords<float[]> bagOfVisualWords = new BagOfVisualWords<>(hardAssigner);
      SparseIntFV featureVector = bagOfVisualWords.aggregateVectorsRaw(patches);
      return featureVector.asDoubleFV();
    }

    /**
     * Extract a list of all the patches from the given image
     *
     * @param image             Image to extract patches from
     * @param patchSize         Size, both width and height, of each patch
     * @param patchDifference   Pixel difference between each patch
     * @param normalisePatches  Whether patches should be normalised
     * @param meanCentrePatches Whether patches should be mean-centred
     * @return List of all patches in the image
     */
    public static List<float[]> getPatchesFromImage(
            FImage image,
            int patchSize,
            int patchDifference,
            boolean normalisePatches,
            boolean meanCentrePatches) {
      List<float[]> patches = new ArrayList<>();
      int rightPixel = 0;
      int bottomPixel = patchSize - 1;

      while (bottomPixel < image.getHeight()) {
        rightPixel = patchSize - 1;
        while (rightPixel < image.getWidth()) {
          float[] patch = new float[patchSize * patchSize];
          int pixelCount = 0;
          for (int x = (bottomPixel - patchSize + 1); x <= bottomPixel; x++) {
            for (int y = (rightPixel - patchSize + 1); y <= rightPixel; y++) {
              patch[pixelCount] = image.pixels[x][y];
              pixelCount++;
            }
          }
          // Optional mean centring and normalisation
          if (meanCentrePatches) {
            meanCentrePatch(patch);
          }
          if (normalisePatches) {
            normalisePatch(patch);
          }
          patches.add(patch);
          rightPixel += patchDifference;
        }
        bottomPixel += patchDifference;
      }

      return patches;
    }

    /**
     * Normalise a pixel patch
     *
     * @param patch Patch to be normalised
     */
    private static void normalisePatch(float[] patch) {
      Float minimum = Collections.min(Arrays.asList(ArrayUtils.toObject(patch)));
      Float maximum = Collections.max(Arrays.asList(ArrayUtils.toObject(patch)));
      for (int i = 0; i < patch.length; i++) {
        patch[i] = (255f / (maximum - minimum)) * (patch[i] - minimum);
      }
    }

    /**
     * Mean-centre a pixel patch
     *
     * @param patch Patch to be mean-centred
     */
    private static void meanCentrePatch(float[] patch) {
      // Calculate mean pixel value
      float total = 0;
      for (float pixel : patch) {
        total += pixel;
      }
      float mean = total / patch.length;
      // Subtract mean from each pixel
      for (int i = 0; i < patch.length; i++) {
        patch[i] = patch[i] - mean;
      }
    }
  }
}
