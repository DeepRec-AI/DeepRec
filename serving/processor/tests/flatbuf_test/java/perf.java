import com.google.flatbuffers.FlatBufferBuilder;
import com.aliyun.openservices.eas.predict.request.TFRequest;
import com.aliyun.openservices.eas.predict.request.TFDataType;
import com.aliyun.openservices.eas.predict.proto.PredictProtos.PredictRequest;
import com.aliyun.openservices.eas.predict.proto.PredictProtos.ArrayProto;
import eas.ContentType;
import eas.DoubleContentType;
import eas.FloatContentType;
import eas.Int64ContentType;
import eas.IntContentType;
import eas.PredictRequest2;
import eas.ShapeType;

import java.util.*;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;


import shade.protobuf.InvalidProtocolBufferException;

public class perf {
  static boolean PRINT_DATA = false;
  static boolean PRINT_FB = false;
  static boolean PRINT_PB = false;
  static boolean WRITE_TO_FILE = false;

  static int DIM_0 = 2;
  static int DIM_1 = 1000;

  // for string type
  static int STR_LEN = 8; // 8, 16, 32

  // Input tensor num
  static int FLOAT_COUNT = 10;
  static int LONG_COUNT = 10;
  static int DOUBLE_COUNT = 10;
  static int INT_COUNT = 10;
  static int STRING_COUNT = 0; 
  static int TOTAL_COUNT =
      FLOAT_COUNT + LONG_COUNT +
      DOUBLE_COUNT + INT_COUNT +
      STRING_COUNT;

  // testing count
  static int TESTING_COUNT = 100;

  public static String strRand(int len) {
    String str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    Random random = new Random();
    StringBuffer sb = new StringBuffer();
    for(int i = 0; i < len; i++) {
      int number = random.nextInt(62);
      sb.append(str.charAt(number));
    }

    return sb.toString();
  }

  public static void prepareData(String sigName, String[] inputNames,
                                 int[] dataTypes, long[][] shapes,
                                 float[][] floatContent, long[][] longContent,
                                 double[][] doubleContent, int[][] intContent,
                                 String[][] stringContent, String[] fetchNames) {
    // sigName
    sigName = "default_serving";

    // fetch names/data types/shapes
    String baseName =
      "input_from_feature_columns/fm_10169_embedding/const";
    for (int i = 0; i < TOTAL_COUNT; ++i) {
      inputNames[i] = baseName + "_" + String.valueOf(i);

      dataTypes[i] = i;

      shapes[i][0] = DIM_0;
      shapes[i][1] = DIM_1;
    }

    // float
    for (int i = 0; i < FLOAT_COUNT; ++i) {
      float startNum = (float)1.22341;
      float incrStep = (float)0.67176;
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        floatContent[i][x] = startNum;
        startNum += incrStep;
      }
    }
  
    // long
    for (int i = 0; i < LONG_COUNT; ++i) {
      long startNum = 223;
      long incrStep = 76;
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        longContent[i][x] = startNum;
        startNum += incrStep;
      }
    }

    // double
    for (int i = 0; i < DOUBLE_COUNT; ++i) {
      double startNum = 223.263763;
      double incrStep = 76.265361;
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        doubleContent[i][x] = startNum;
        startNum += incrStep;
      }
    }

    // int
    for (int i = 0; i < INT_COUNT; ++i) {
      int startNum = 300;
      int incrStep = 1;
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        intContent[i][x] = startNum;
        startNum += incrStep;
      }
    }

    // String
    String[] strTmp = new String[DIM_0 * DIM_1];
    for (int x = 0; x < DIM_0 * DIM_1; ++x) {
      strTmp[x] = strRand(STR_LEN);
    }
    for (int i = 0; i < STRING_COUNT; ++i) {
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        stringContent[i][x] = strTmp[x];
      }
    }

    // fetch names
    fetchNames[0] = "fetch_0";
    fetchNames[1] = "fetch_1";
    fetchNames[2] = "fetch_2";
  }

  public static void printData(String sigName, String[] inputNames,
                               int[] dataTypes, long[][] shapes,
                               float[][] floatContent, long[][] longContent,
                               double[][] doubleContent, int[][] intContent,
                               String[][] stringContent, String[] fetchNames) {
    System.out.println("signatureName: " + sigName);

    for (int i = 0; i < TOTAL_COUNT; ++i) {
      System.out.println("int_name: " + inputNames[i]);
    }

    for (int i = 0; i < TOTAL_COUNT; ++i) {
      System.out.println("data_type: " + dataTypes[i]);
    }

    for (int i = 0; i < TOTAL_COUNT; ++i) {
      System.out.println("shape: [" + shapes[i][0] + ", " + shapes[i][1] + "]");
    }

    System.out.println("\nfloat content:\n");
    // float
    for (int i = 0; i < FLOAT_COUNT; ++i) {
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        System.out.println(floatContent[i][x]);
      }
    }
  
    System.out.println("\nlong content:\n");
    // long
    for (int i = 0; i < LONG_COUNT; ++i) {
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        System.out.println(longContent[i][x]);
      }
    }

    System.out.println("\ndouble content:\n");
    // double
    for (int i = 0; i < DOUBLE_COUNT; ++i) {
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        System.out.println(doubleContent[i][x]);
      }
    }

    System.out.println("\nint content:\n");
    // int
    for (int i = 0; i < INT_COUNT; ++i) {
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        System.out.println(intContent[i][x]);
      }
    }

    System.out.println("\nstring content:\n");
    // String
    for (int i = 0; i < STRING_COUNT; ++i) {
      for (int x = 0; x < DIM_0 * DIM_1; ++x) {
        System.out.println(stringContent[i][x]);
      }
    }

    // fetch names
    System.out.println("fetch_name: " + fetchNames[0]);
    System.out.println("fetch_name: " + fetchNames[1]);
    System.out.println("fetch_name: " + fetchNames[2]);
  }

  public static void printFB(ByteBuffer buf) {
    PredictRequest2 recvReq = PredictRequest2.getRootAsPredictRequest2(buf);

    System.out.println("signatureName = " + recvReq.signatureName());

    int idx = 0;
    for (int i = 0; i < recvReq.feedNamesLength(); i++) {
      System.out.println("#" + idx + "_feed_name = " + recvReq.feedNames(i));
      idx++;
    }

    idx = 0;
    for (int i = 0; i < recvReq.typesLength(); i++) {
      System.out.println("#" + idx + "_data_type = " + recvReq.types(i));
      idx++;
    }

    idx = 0;
    for (int i = 0; i < recvReq.shapesLength(); i++) {
      // perf shapesVector performance
      eas.ShapeType st = recvReq.shapes(i);
      long[] dim = new long[st.dimLength()];
      for (int j = 0; j < st.dimLength(); j++) {
        dim[j] = st.dim(j);
      }
      System.out.print("#" + idx + "_shape = [");
      for (int j = 0; j < dim.length; ++j) {
        System.out.print(dim[j] + ", ");
      }
      System.out.println("]");
      idx++;
    }

    idx = 0;
    for (int i = 0; i < recvReq.floatContentLength(); i++) {
      System.out.println("float nums: #" + i);
      eas.FloatContentType fct = recvReq.floatContent(i);
      for (int j = 0; j < fct.contentLength(); j++) {
        System.out.println("#" + idx + "_ = " + fct.content(j));
        idx++;
      }
    }

    idx = 0;
    for (int i = 0; i < recvReq.i64ContentLength(); i++) {
      System.out.println("long nums: #" + i);
      eas.Int64ContentType ct = recvReq.i64Content(i);
      for (int j = 0; j < ct.contentLength(); j++) {
        System.out.println("#" + idx + "_ = " + ct.content(j));
        idx++;
      }
    }

    idx = 0;
    for (int i = 0; i < recvReq.dContentLength(); i++) {
      System.out.println("double nums: #" + i);
      eas.DoubleContentType ct = recvReq.dContent(i);
      for (int j = 0; j < ct.contentLength(); j++) {
        System.out.println("#" + idx + "_ = " + ct.content(j));
        idx++;
      }
    }

    idx = 0;
    for (int i = 0; i < recvReq.iContentLength(); i++) {
      System.out.println("int nums: #" + i);
      eas.IntContentType ct = recvReq.iContent(i);
      for (int j = 0; j < ct.contentLength(); j++) {
        System.out.println("#" + idx + "_ = " + ct.content(j));
        idx++;
      }
    }

    idx = 0;
    System.out.println("string content len: ");
    int[] lens = new int[recvReq.stringContentLenLength()];
    for (int i = 0; i < recvReq.stringContentLenLength(); i++) {
      lens[i] = recvReq.stringContentLen(i);
      System.out.println("#" + idx + " = " + lens[i]);
      idx++;
    }

    idx = 0;
    int offset = 0;
    System.out.println("string content: ");
    for (int i = 0; i < recvReq.stringContentLength(); i++) {
      int count = lens[offset++];
      int cur = 0;
      String curStr = recvReq.stringContent(i);
      for (int j = 0; j < count; j++) {
        System.out.println("#" + idx + "_string: " + curStr.substring(cur, cur + lens[offset]));
        cur += lens[offset++];
        idx++;
      }
    }
  }

  public static void encodeByFlatBuffer(FlatBufferBuilder fbb,
                                        String signatureName, String[] inputNames,
                                        int[] dataTypes, long[][] shapes,
                                        float[][] floatContent, long[][] longContent,
                                        double[][] doubleContent, int[][] intContent,
                                        String[][] stringContent, String[] fetchNames) {

    // --- signatureName
    int fsigNameOffset = fbb.createString(signatureName);

    // --- inputNames
    List<Integer> tmpInputNamesOffset = new ArrayList<Integer>(10);
    for (int i = 0; i < inputNames.length; ++i) {
      tmpInputNamesOffset.add(fbb.createString(inputNames[i]));
    }
    int[] inputNamesOffsets = new int[tmpInputNamesOffset.size()];
    for (int i = 0; i < tmpInputNamesOffset.size(); ++i) {
      inputNamesOffsets[i] = tmpInputNamesOffset.get(i);
    }
    int finputNamesOffset =
        PredictRequest2.createFeedNamesVector(fbb, inputNamesOffsets);

    // --- data types
    List<Integer> tmpTypes = new ArrayList<Integer>(10);
    for (int i = 0; i < dataTypes.length; ++i) {
      tmpTypes.add(dataTypes[i]);
    }
    fbb.startVector(4, tmpTypes.size(), 4);
    for (int i = tmpTypes.size() - 1; i >= 0; i--) {
      fbb.addInt(tmpTypes.get(i));
    }
    int fdataTypesOffset = fbb.endVector();

    // --- shapes
    List<Integer> tmpShapesOffset = new ArrayList<Integer>(10);
    for (int i = 0; i < shapes.length; i++) {
      // user pass long[] everytime
      // public void addFeed(String inputName, TFDataType dataType,
      //                     long[] shape, float[] content);
      int o = ShapeType.createDimVector(fbb, shapes[i]);
      ShapeType.startShapeType(fbb);
      ShapeType.addDim(fbb, o);
      int endOffset = ShapeType.endShapeType(fbb);
      tmpShapesOffset.add(endOffset);
    }
    fbb.startVector(4, tmpShapesOffset.size(), 4);
    for (int i = tmpShapesOffset.size() - 1; i >= 0; i--) {
      fbb.addOffset(tmpShapesOffset.get(i));
    }
    int fshapesOffsets = fbb.endVector();

    // --- float content
    List<Integer> tmpFloatOffset = new ArrayList<Integer>(10);
    for (int i = 0; i < floatContent.length; ++i) {
      int o = FloatContentType.createContentVector(fbb, floatContent[i]);
      FloatContentType.startFloatContentType(fbb);
      FloatContentType.addContent(fbb, o);
      int endOffset = FloatContentType.endFloatContentType(fbb);
      tmpFloatOffset.add(endOffset);
    }
    fbb.startVector(4, tmpFloatOffset.size(), 4);
    for (int i = tmpFloatOffset.size() - 1; i >= 0; i--) {
      fbb.addOffset(tmpFloatOffset.get(i));
    }
    int ffcontentOffset = fbb.endVector();

    // --- long content
    List<Integer> tmpLongOffset = new ArrayList<Integer>(10);
    for (int i = 0; i < longContent.length; ++i) {
      int o = Int64ContentType.createContentVector(fbb, longContent[i]);
      Int64ContentType.startInt64ContentType(fbb);
      Int64ContentType.addContent(fbb, o);
      int endOffset = Int64ContentType.endInt64ContentType(fbb);
      tmpLongOffset.add(endOffset);
    }
    fbb.startVector(4, tmpLongOffset.size(), 4);
    for (int i = tmpLongOffset.size() - 1; i >= 0; i--) {
      fbb.addOffset(tmpLongOffset.get(i));
    }
    int flcontentOffset = fbb.endVector();

    // --- double content
    List<Integer> tmpDoubleOffset = new ArrayList<Integer>(10);
    for (int i = 0; i < doubleContent.length; ++i) {
      int o = DoubleContentType.createContentVector(fbb, doubleContent[i]);
      DoubleContentType.startDoubleContentType(fbb);
      DoubleContentType.addContent(fbb, o);
      int endOffset = DoubleContentType.endDoubleContentType(fbb);
      tmpDoubleOffset.add(endOffset);
    }
    fbb.startVector(4, tmpDoubleOffset.size(), 4);
    for (int i = tmpDoubleOffset.size() - 1; i >= 0; i--) {
      fbb.addOffset(tmpDoubleOffset.get(i));
    }
    int fdcontentOffset = fbb.endVector();

    // --- int content
    List<Integer> tmpIntOffset = new ArrayList<Integer>(10);
    for (int i = 0; i < intContent.length; ++i) {
      int o = IntContentType.createContentVector(fbb, intContent[i]);
      IntContentType.startIntContentType(fbb);
      IntContentType.addContent(fbb, o);
      int endOffset = IntContentType.endIntContentType(fbb);
      tmpIntOffset.add(endOffset);
    }
    fbb.startVector(4, tmpIntOffset.size(), 4);
    for (int i = tmpIntOffset.size() - 1; i >= 0; i--) {
      fbb.addOffset(tmpIntOffset.get(i));
    }
    int ficontentOffset = fbb.endVector();

    // --- string len
    List<Integer> strLenArray = new ArrayList<Integer>(1000);
    List<Integer> stringAggregateBufsOffset = new ArrayList<Integer>(10);
    int totalCount = 0;
    for (int i = 0; i < stringContent.length; ++i) {
      StringBuilder tmpStr = new StringBuilder(
          stringContent[i][0].length() * stringContent[i].length);
      int offset = 0;
      strLenArray.add(stringContent[i].length);
      for (int j = 0; j < stringContent[i].length; j++) {
        strLenArray.add(stringContent[i][j].length());
        tmpStr.append(stringContent[i][j]);
      }
      totalCount += 1;
      totalCount += stringContent[i].length;

      stringAggregateBufsOffset.add(fbb.createString(tmpStr.toString()));
    }
    fbb.startVector(4, totalCount, 4);
    for (int i = strLenArray.size() - 1; i >= 0; i--) {
      fbb.addInt(strLenArray.get(i));
    }
    int fsStringLenOffset = fbb.endVector();

    // --- string content
    fbb.startVector(4, stringAggregateBufsOffset.size(), 4);
    for (int i = stringAggregateBufsOffset.size() - 1; i >= 0; i--) {
      fbb.addOffset(stringAggregateBufsOffset.get(i));
    }
    int fscontentOffset = fbb.endVector();

    // fetch names
    int[] fetchNamesOffset = new int[fetchNames.length];
    for (int i = 0; i < fetchNames.length; ++i) {
      fetchNamesOffset[i] = fbb.createString(fetchNames[i]);
    }
    int ffetchNamesOffset =
        PredictRequest2.createFeedNamesVector(fbb, fetchNamesOffset);

    // --- wrap to flatbuffer
    PredictRequest2.startPredictRequest2(fbb);
    PredictRequest2.addSignatureName(fbb, fsigNameOffset);
    PredictRequest2.addFeedNames(fbb, finputNamesOffset);
    PredictRequest2.addTypes(fbb, fdataTypesOffset);
    PredictRequest2.addShapes(fbb, fshapesOffsets);
    PredictRequest2.addFloatContent(fbb, ffcontentOffset);
    PredictRequest2.addI64Content(fbb, flcontentOffset);
    PredictRequest2.addDContent(fbb, fdcontentOffset);
    PredictRequest2.addIContent(fbb, ficontentOffset);
    PredictRequest2.addStringContentLen(fbb, fsStringLenOffset);
    PredictRequest2.addStringContent(fbb, fscontentOffset);
    PredictRequest2.addFetchNames(fbb, ffetchNamesOffset);
    int endOffset = PredictRequest2.endPredictRequest2(fbb);
    fbb.finish(endOffset);
  }

  public static void printPB(byte[] buf) {
    try {
      PredictRequest req = PredictRequest.parseFrom(buf);
      System.out.println("pb signatureName: " + req.getSignatureName());
      int idx = 0;
      for (int i = 0; i < req.getOutputFilterCount(); i++) {
        System.out.println("pb fetch_name: #" + idx + "_" + req.getOutputFilter(i));
        idx++;
      }
      for (String key : req.getInputsMap().keySet()) {
        System.out.println("Key = " + key); 
      }
      idx = 0;
      for (ArrayProto val : req.getInputsMap().values()) {
        // float
        for (int i = 0; i < val.getFloatValCount(); i++) {
          System.out.println("float #" + idx + "_" + val.getFloatVal(i));
          idx++;
        }
        // long
        idx = 0;
        for (int i = 0; i < val.getInt64ValCount(); i++) {
          System.out.println("long #" + idx + "_" + val.getInt64Val(i));
          idx++;
        }
        // double
        for (int i = 0; i < val.getDoubleValCount(); i++) {
          System.out.println("double #" + idx + "_" + val.getDoubleVal(i));
          idx++;
        }
        // int
        for (int i = 0; i < val.getIntValCount(); i++) {
          System.out.println("int #" + idx + "_" + val.getIntVal(i));
          idx++;
        }
        // string
        for (int i = 0; i < val.getStringValCount(); i++) {
          System.out.println("string #" + idx + "_" + val.getStringVal(i).toStringUtf8());
          idx++;
        }
      }
    } catch (InvalidProtocolBufferException e) {
      e.printStackTrace();
    } catch (NullPointerException e) {
      e.printStackTrace();
    }
  }

  public static void main(String[] args) {

    String signatureName = "default_serving";
    String[] inputNames = new String[TOTAL_COUNT];
    int[] dataTypes = new int[TOTAL_COUNT];
    long[][] shapes = new long[TOTAL_COUNT][2];
    float[][] floatContent = new float[FLOAT_COUNT][DIM_0*DIM_1];
    long[][] longContent = new long[LONG_COUNT][DIM_0*DIM_1];
    double[][] doubleContent = new double[DOUBLE_COUNT][DIM_0*DIM_1];
    int[][] intContent = new int[INT_COUNT][DIM_0*DIM_1];
    String[][] stringContent = new String[STRING_COUNT][DIM_0*DIM_1];
    String[] fetchNames = new String[3];

    prepareData(signatureName, inputNames, dataTypes, shapes,
                floatContent, longContent, doubleContent,
                intContent, stringContent, fetchNames);

    if (PRINT_DATA) {
      printData(signatureName, inputNames, dataTypes, shapes,
                floatContent, longContent, doubleContent,
                intContent, stringContent, fetchNames);
    }

    // ================ flatbuffer ===================
    double startFBMillis = System.currentTimeMillis();
    int fbLen = 0;
    for (int x = 0; x < TESTING_COUNT; ++x) {
      // encode
      FlatBufferBuilder fbb = new FlatBufferBuilder(20000);

      // no copy here, convert user content to
      // flatbuffer data directly.
      encodeByFlatBuffer(fbb, signatureName, inputNames, dataTypes,
                         shapes, floatContent, longContent,
                         doubleContent, intContent,
                         stringContent, fetchNames);
  
      byte[] buf = fbb.sizedByteArray();
      fbLen = buf.length;

      if (WRITE_TO_FILE) {
        try {
          File file = new File("/tmp/request_buf.bin");
          FileOutputStream fop = new FileOutputStream(file);
          fop.write(buf);
          fop.flush();
          fop.close();
        } catch (IOException e) {
          e.printStackTrace();
        }
      }

      if (PRINT_FB) {
        printFB(ByteBuffer.wrap(buf));
      }
    }
    double endFBMillis = System.currentTimeMillis();

    // ================ protobuffer ===================
    double startPBMillis = System.currentTimeMillis();
    int pbLen = 0;
    for (int x = 0; x < TESTING_COUNT; ++x) {
      TFRequest pbReq = new TFRequest();

      // encode
      pbReq.setSignatureName(signatureName);
      for (int i = 0; i < fetchNames.length; i++) {
        pbReq.addFetch(fetchNames[i]);
      }
      int offset = 0;
      for (int i = 0; i < floatContent.length; i++) {
        pbReq.addFeed(inputNames[offset], TFDataType.DT_FLOAT,
                       shapes[offset], floatContent[i]);
        offset++;
      }
      for (int i = 0; i < longContent.length; i++) {
        pbReq.addFeed(inputNames[offset], TFDataType.DT_INT64,
                       shapes[offset], longContent[i]);
        offset++;
      }
      for (int i = 0; i < doubleContent.length; i++) {
        pbReq.addFeed(inputNames[offset], TFDataType.DT_DOUBLE,
                       shapes[offset], doubleContent[i]);
        offset++;
      }
      for (int i = 0; i < intContent.length; i++) {
        pbReq.addFeed(inputNames[offset], TFDataType.DT_INT32,
                       shapes[offset], intContent[i]);
        offset++;
      }
      for (int i = 0; i < stringContent.length; i++) {
        pbReq.addFeed(inputNames[offset], TFDataType.DT_STRING,
                       shapes[offset], stringContent[i]);
        offset++;
      }

      byte[] pbBuf = pbReq.getRequest().toByteArray();
      pbLen = pbBuf.length;

      // decode & print
      if (PRINT_PB) {
        printPB(pbBuf);
      }
    }

    double endPBMillis = System.currentTimeMillis();

    // ================ print_result ===================

    System.out.println("===================> fb cost: " +
                       (endFBMillis - startFBMillis)/TESTING_COUNT +
                       ", len = " + fbLen);
    System.out.println("===================> pb cost: " +
                       (endPBMillis - startPBMillis)/TESTING_COUNT +
                       ", len = " + pbLen);
  }
}

