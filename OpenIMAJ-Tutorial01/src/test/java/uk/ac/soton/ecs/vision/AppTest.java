package uk.ac.soton.ecs.vision;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.openimaj.image.FImage;
import uk.ac.soton.ecs.vision.DenselySampledPixelPatchesClassifier.DenselySampledPixelPatchesExtractor;

/**
 * Unit test for simple App.
 */
public class AppTest {
    /**
     * Rigourous Test :-)
     */
//	@Test
//  public void testMeanCentringPatch() {
//	  //float[] patch = {7, 30, 2, 57, 33, 99, 40, 22, 29, 22, 15, 43, 33, 50, 80, 5};
//    float[] patch = {7, 30, 2, 57}; // mean=24
//		DenselySampledPixelPatchesExtractor.meanCentrePatch(patch);
//	  float[] expectedResult = {-17, 6, -22, 33};
//	  assertArrayEquals(expectedResult, patch, 0.0001f);
//	}
//
//	@Test
//	public void testNormalisingPatch() {
//		float[] patch = {7, 30, 2, 57};
//		DenselySampledPixelPatchesExtractor.normalisePatch(patch);
//		float[] expectedResult = {(255f/11), (1428f/11), 0, 255};
//		assertArrayEquals(expectedResult, patch, 0.0001f);
//	}
//
//	@Test
//	public void testGetPatches() {
//		float[][] pixels = {{0,1,2,3,4,5,6,7,8,9,9.5f,9.75f}, {10,11,12,13,14,15,16,17,18,19,19.5f,19.75f}, {20,21,22,23,24,25,26,27,28,29,29.5f,29.75f},
//				{30,31,32,33,34,35,36,37,38,39,39.5f,39.75f}, {40,41,42,43,44,45,46,47,48,49,49.5f,49.75f}, {50,51,52,53,54,55,56,57,58,59,59.5f,59.75f},
//				{60,61,62,63,64,65,66,67,68,69,69.5f,69.75f}, {70,71,72,73,74,75,76,77,78,79,79.5f,79.75f}, {80,81,82,83,84,85,86,87,88,89,89.5f,89.75f},
//				{90,91,92,93,94,95,96,97,98,99,99.5f,99.75f}, {100,101,102,103,104,105,106,107,108,109,109.5f,109.75f}, {110,111,112,113,114,115,116,117,118,119,119.5f,119.75f}};
//		FImage image = new FImage(pixels);
//		List<float[]> patches = DenselySampledPixelPatchesExtractor
//				.getPatchesFromImage(image, 8, 4, false, false);
//		boolean allPatchesSame = true;
//		float[] patch1 = {0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17,20,21,22,23,24,25,26,27,30,31,32,33,34,35,36,37,40,41,42,43,44,45,46,47,50,51,52,53,54,55,56,57,60,61,62,63,64,65,66,67,70,71,72,73,74,75,76,77};
//		float[] patch2 = {4,5,6,7,8,9,9.5f,9.75f,14,15,16,17,18,19,19.5f,19.75f,24,25,26,27,28,29,29.5f,29.75f,34,35,36,37,38,39,39.5f,39.75f,44,45,46,47,48,49,49.5f,49.75f,54,55,56,57,58,59,59.5f,59.75f,
//				64,65,66,67,68,69,69.5f,69.75f,74,75,76,77,78,79,79.5f,79.75f};
//		float[] patch3 = {40,41,42,43,44,45,46,47,50,51,52,53,54,55,56,57,60,61,62,63,64,65,66,67,70,71,72,73,74,75,76,77,80,81,82,83,84,85,86,87,90,91,92,93,94,95,96,97,100,101,102,103,104,105,106,107,110,111,112,113,114,115,116,117};
//		float[] patch4 = {44,45,46,47,48,49,49.5f,49.75f,54,55,56,57,58,59,59.5f,59.75f,64,65,66,67,68,69,69.5f,69.75f,74,75,76,77,78,79,79.5f,79.75f,84,85,86,87,88,89,89.5f,89.75f,94,95,96,97,98,99,99.5f,99.75f,104,105,106,107,108,109,109.5f,109.75f,114,115,116,117,118,119,119.5f,119.75f,};
//		List<float[]> expectedPatches = new ArrayList<>();
//		expectedPatches.add(patch1);
//		expectedPatches.add(patch2);
//		expectedPatches.add(patch3);
//		expectedPatches.add(patch4);
//
//		for (int x = 0; x < patches.size(); x++) {
//			for (int i = 0; i < patches.get(x).length; i++) {
//				System.out.println(patches.get(x)[i] + "   " + expectedPatches.get(x)[i]);
//				if (patches.get(x)[i] != expectedPatches.get(x)[i]) {
//					allPatchesSame = false;
//				}
//			}
//			System.out.println("-------------------------------------------------------------------------");
//		}
//		assertTrue(allPatchesSame);
//	}
}
