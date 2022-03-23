#!/usr/bin/python

import string
import random
import array
from copy import copy, deepcopy
import time
import numpy as np
import flatbuffers
import eas.FloatContentType as FloatContentType
import eas.ContentType as ContentType
import eas.ShapeType as ShapeType
import eas.Int64ContentType as Int64ContentType
import eas.PredictRequest as PredictRequest
import eas.DoubleContentType as DoubleContentType
import eas.IntContentType as IntContentType
from pb.tf_request import TFRequest
from pb import tf_request_pb2 as tf_pb

def str_rand(length) :
  seed = "1234567890abcdefghijklmnopqrstuvwxyz" \
         "ABCDEFGHIJKLMNOPQRSTUVWXYZ_-"
  sa = []
  for i in range(length):
    sa.append(random.choice(seed))

  return ''.join(sa)

def main():
  # prepare data -------------------------

  DEBUG = False
  TESTING_COUNT = 100

  float_count = 10
  int64_count = 10
  double_count = 10
  int_count = 10
  str_count = 10
  dim0 = 2
  dim1 = 1000
  string_content_len = 8

  total_count = float_count + \
                int64_count + \
                double_count + \
                int_count + \
                str_count

  # signature name
  signature_name = "default_serving"
  # input names
  input_names = []
  for i in range(0, total_count):
    input_names.append("input_name_" + str(i))
  # data types
  data_types = []
  for i in range(0, total_count):
    data_types.append(i)
  # shapes
  shapes = []
  for i in range(0, total_count):
    each_shape = []
    each_shape.append(dim0)
    each_shape.append(dim1)
    shapes.append(each_shape)
  # contents

  # --- float ----
  contents = []
  start_num = 1.22341
  incr_step = 0.67176
  each_contents = []
  each_content = array.array('f')
  for i in range(dim0):
    for j in range(dim1):
      each_content.append(start_num)
      start_num = start_num + incr_step
  each_content = np.array(each_content, dtype=np.float32)
  for i in range(float_count):
    each_contents.append(deepcopy(each_content))
    each_contents[i][0] = float(i)
    contents.append(each_contents[i])

  # --- int64 ----
  i64_contents = []
  i64_start_num = 1241
  i64_incr_step = 715
  i64_each_contents = []
  i64_each_content = array.array('l')
  for i in range(dim0):
    for j in range(dim1):
      i64_each_content.append(i64_start_num)
      i64_start_num = i64_start_num + i64_incr_step
  i64_each_content = np.array(i64_each_content, dtype=np.int64)
  for i in range(int64_count):
    i64_each_contents.append(deepcopy(i64_each_content))
    i64_each_contents[i][0] = i
    i64_contents.append(i64_each_contents[i])

  # --- double ----
  d_contents = []
  d_start_num = 1.22341
  d_incr_step = 0.67176
  d_each_contents = []
  d_each_content = array.array('d')
  for i in range(dim0):
    for j in range(dim1):
      d_each_content.append(d_start_num)
      d_start_num = d_start_num + d_incr_step
  d_each_content = np.array(d_each_content, dtype=np.float64)
  for i in range(double_count):
    d_each_contents.append(deepcopy(d_each_content))
    d_each_contents[i][0] = float(i)
    d_contents.append(d_each_contents[i])

  # --- int32 ----
  i_contents = []
  i_start_num = 12
  i_incr_step = 15
  i_each_contents = []
  i_each_content = array.array('l')
  for i in range(dim0):
    for j in range(dim1):
      i_each_content.append(i_start_num)
      i_start_num = i_start_num + i_incr_step
  i_each_content = np.array(i_each_content, dtype=np.int32)
  for i in range(int_count):
    i_each_contents.append(deepcopy(i_each_content))
    i_each_contents[i][0] = i
    i_contents.append(i_each_contents[i])

  # string content
  s_contents = []
  s_each_content = [] # Dim0 * Dim1 * string
  for i in range(dim0):
    for j in range(dim1):
      s_each_content.append(str_rand(string_content_len))
  for i in range(str_count):
    s_contents.append(s_each_content)

  # output/fetch name
  output_name = ["fetch_0", "fetch_1", "fetch_2"]
  
  #print (input_names)
  #print (len(input_names))
  #print (data_types)
  #print (shapes)
  #print (each_content)
  #print (contents)
  #print (i64_each_contents)
  #print (d_each_contents)
  #print (s_each_content)
  #print (s_contents)
  #print (output_name)

  # encoding fb --------------------

  stime = time.time()

  for test_count in range(TESTING_COUNT):
    builder = flatbuffers.Builder(0)

    # signature name content

    fsig_name = builder.CreateString(signature_name)

    # input name content

    finput_names = []
    for name in input_names:
      tmp_name = builder.CreateString(name)
      finput_names.append(tmp_name)
    PredictRequest.PredictRequestStartFeedNamesVector(builder, len(finput_names))
    # WHY?
    # reverse order?
    for name in finput_names[::-1]:
      builder.PrependUOffsetTRelative(name)
    finput_names_offset = builder.EndVector(len(finput_names))
 
    # type content

    PredictRequest.PredictRequestStartTypesVector(builder, len(data_types))
    for t in data_types[::-1]:
      builder.PrependInt32(t)
    fdata_types_offset = builder.EndVector(len(data_types))
  
    # shape content

    shape_offsets = []
    for s in shapes[::-1]:
      ShapeType.ShapeTypeStartDimVector(builder, len(s))
      for d in s[::-1]:
        builder.PrependInt64(d)
      shape_offset = builder.EndVector(len(s))
  
      ShapeType.ShapeTypeStart(builder)
      ShapeType.ShapeTypeAddDim(builder, shape_offset)
      shape_type_offset = ShapeType.ShapeTypeEnd(builder)
  
      shape_offsets.append(shape_type_offset)
  
    PredictRequest.PredictRequestStartShapesVector(builder, len(shape_offsets))
    for offset in shape_offsets[::-1]:
      builder.PrependUOffsetTRelative(offset)
    fshape_offsets = builder.EndVector(len(shape_offsets))
  
    # float content

    float_content_offsets = []
    for c in contents:
      #FloatContentType.FloatContentTypeStartContentVector(builder, len(c))
      #for num in c[::-1]:
      #  builder.PrependFloat32(num)
      #c_offset = builder.EndVector(len(c))
      c_offset = builder.CreateNumpyVector(c)

      FloatContentType.FloatContentTypeStart(builder)
      FloatContentType.FloatContentTypeAddContent(builder, c_offset)
      content_offset = FloatContentType.FloatContentTypeEnd(builder)
  
      float_content_offsets.append(content_offset)
 
    PredictRequest.PredictRequestStartFloatContentVector(builder, len(float_content_offsets))
    for offset in float_content_offsets[::-1]:
      builder.PrependUOffsetTRelative(offset)
    ffloat_content_offsets = builder.EndVector(len(float_content_offsets))

    # int64 content

    i64_content_offsets = []
    for c in i64_contents:
      #Int64ContentType.Int64ContentTypeStartContentVector(builder, len(c))
      #for num in c[::-1]:
      #  builder.PrependInt64(num)
      #c_offset = builder.EndVector(len(c))
      c_offset = builder.CreateNumpyVector(c)

      Int64ContentType.Int64ContentTypeStart(builder)
      Int64ContentType.Int64ContentTypeAddContent(builder, c_offset)
      content_offset = Int64ContentType.Int64ContentTypeEnd(builder)
  
      i64_content_offsets.append(content_offset)

    PredictRequest.PredictRequestStartI64ContentVector(builder, len(i64_content_offsets))
    for offset in i64_content_offsets[::-1]:
      builder.PrependUOffsetTRelative(offset)
    fi64_content_offsets = builder.EndVector(len(i64_content_offsets))

    # double content

    d_content_offsets = []
    for c in d_contents:
      #DoubleContentType.DoubleContentTypeStartContentVector(builder, len(c))
      #for num in c[::-1]:
      #  builder.PrependFloat64(num)
      #c_offset = builder.EndVector(len(c))
      c_offset = builder.CreateNumpyVector(c)

      DoubleContentType.DoubleContentTypeStart(builder)
      DoubleContentType.DoubleContentTypeAddContent(builder, c_offset)
      content_offset = DoubleContentType.DoubleContentTypeEnd(builder)
  
      d_content_offsets.append(content_offset)

    PredictRequest.PredictRequestStartDContentVector(builder, len(d_content_offsets))
    for offset in d_content_offsets[::-1]:
      builder.PrependUOffsetTRelative(offset)
    fd_content_offsets = builder.EndVector(len(d_content_offsets))

    # int content

    i_content_offsets = []
    for c in i_contents:
      #IntContentType.IntContentTypeStartContentVector(builder, len(c))
      #for num in c[::-1]:
      #  builder.PrependInt32(num)
      #c_offset = builder.EndVector(len(c))
      c_offset = builder.CreateNumpyVector(c)

      IntContentType.IntContentTypeStart(builder)
      IntContentType.IntContentTypeAddContent(builder, c_offset)
      content_offset = IntContentType.IntContentTypeEnd(builder)
  
      i_content_offsets.append(content_offset)

    PredictRequest.PredictRequestStartIContentVector(builder, len(i_content_offsets))
    for offset in i_content_offsets[::-1]:
      builder.PrependUOffsetTRelative(offset)
    fi_content_offsets = builder.EndVector(len(i_content_offsets))

    # string content/content_len

    # str_content: input0 -> buf0, input1 -> buf1 ...
    # content_len: count, size0, size1 ... size_N; count size0, size1 ... size_N; ...
    s_content_len_array = array.array('l')
    s_content_array = []
    for c in  s_contents:
      s_content_len_array.append(len(c))
      str_join = ""
      for s in c:
        str_join = str_join + s
        s_content_len_array.append(len(s))
      s_content_array.append(str_join)

    ### content_len
    snp_content_len_array = np.array(s_content_len_array, dtype=np.int32)
    fs_content_len_offset = builder.CreateNumpyVector(snp_content_len_array)

    ### str_content
    fs_contents_offsets = []
    fs_contents = []
    for c in s_content_array:
      tmp = builder.CreateString(c)
      fs_contents.append(tmp)
    PredictRequest.PredictRequestStartStringContentVector(builder, len(fs_contents))
    for c in fs_contents[::-1]:
      builder.PrependUOffsetTRelative(c)
    fs_contents_offsets = builder.EndVector(len(fs_contents))

    # output name content

    foutput_names = []
    for name in output_name:
      tmp_name = builder.CreateString(name)
      foutput_names.append(tmp_name)
    PredictRequest.PredictRequestStartFetchNamesVector(builder, len(output_name))
    for name in foutput_names[::-1]:
      builder.PrependUOffsetTRelative(name)
    foutput_names_offset = builder.EndVector(len(output_name))
  
    # encode PredictRequest
    PredictRequest.PredictRequestStart(builder)
    PredictRequest.PredictRequestAddSignatureName(builder, fsig_name)
    PredictRequest.PredictRequestAddFeedNames(builder, finput_names_offset)
    PredictRequest.PredictRequestAddTypes(builder, fdata_types_offset)
    PredictRequest.PredictRequestAddShapes(builder, fshape_offsets)
    PredictRequest.PredictRequestAddFloatContent(builder, ffloat_content_offsets)
    PredictRequest.PredictRequestAddI64Content(builder, fi64_content_offsets)
    PredictRequest.PredictRequestAddDContent(builder, fd_content_offsets)
    PredictRequest.PredictRequestAddIContent(builder, fi_content_offsets)
    PredictRequest.PredictRequestAddStringContentLen(builder, fs_content_len_offset)
    PredictRequest.PredictRequestAddStringContent(builder, fs_contents_offsets)
    PredictRequest.PredictRequestAddFetchNames(builder, foutput_names_offset)
    request = PredictRequest.PredictRequestEnd(builder)
    builder.Finish(request)
  
    buf = builder.Output()
  
  etime = time.time()
  
  t = (etime - stime) / TESTING_COUNT * 1000
  print('fb encode time is {} ms'.format(t), ", size = ", len(buf))

  if DEBUG:
    # decode PredictRequest
    new_req = PredictRequest.PredictRequest.GetRootAsPredictRequest(buf, 0)

    print("SignatureName = ", new_req.SignatureName())
    feed_name_count = new_req.FeedNamesLength()
    for i in range(feed_name_count):
      print("feed_name = ", new_req.FeedNames(i))
    type_count = new_req.TypesLength()
    for i in range(type_count):
      print("type = ", new_req.Types(i))
    shape_count = new_req.ShapesLength();
    for i in range(shape_count):
      shape_i = new_req.Shapes(i)
      print("shape = ", shape_i)
      dim_count = shape_i.DimLength()
      for j in range(dim_count):
        print(shape_i.Dim(j))
    float_content_count = new_req.FloatContentLength()
    for i in range(float_content_count):
      content_i = new_req.FloatContent(i)
      print("float_content = ", content_i)
      num_count = content_i.ContentLength()
      for j in range(num_count):
        print(content_i.Content(j))
    i64_content_count = new_req.I64ContentLength()
    for i in range(i64_content_count):
      content_i = new_req.I64Content(i)
      print("int64_content = ", content_i)
      num_count = content_i.ContentLength()
      for j in range(num_count):
        print(content_i.Content(j))
    d_content_count = new_req.DContentLength()
    for i in range(d_content_count):
      content_i = new_req.DContent(i)
      print("double_content = ", content_i)
      num_count = content_i.ContentLength()
      for j in range(num_count):
        print(content_i.Content(j))
    i_content_count = new_req.IContentLength()
    for i in range(i_content_count):
      content_i = new_req.IContent(i)
      print("int_content = ", content_i)
      num_count = content_i.ContentLength()
      for j in range(num_count):
        print(content_i.Content(j))
    s_content_len_count = new_req.StringContentLenLength()
    s_content_len_array = []
    for i in range(s_content_len_count):
      s_content_len_array.append(new_req.StringContentLen(i))
    s_content_count = new_req.StringContentLength()
    idx = 0
    for i in range(s_content_count):
      each_content = new_req.StringContent(i)
      each_count = s_content_len_array[idx]
      idx = idx + 1
      offset = 0
      for i in range(each_count):
        print("----> content: ", each_content[offset:offset+s_content_len_array[idx]])
        offset = offset + s_content_len_array[idx]
        idx = idx + 1
    fetch_count = new_req.FetchNamesLength()
    for i in range(fetch_count):
      print("fetch_name = ", new_req.FetchNames(i))

    # write buf to file, will be check by cpp programe
    #
    with open("/tmp/request_buf.bin", mode="wb") as f:
      f.write(buf)



  # encoding pb --------------------
  pb_stime = time.time()

  for test_count in range(TESTING_COUNT):
   
    pb_req = TFRequest(signature_name)
    for name in output_name:
      pb_req.add_fetch(name)
    i_offset = 0
    for i in range(float_count):
      pb_req.add_feed(input_names[i_offset], shapes[i_offset], tf_pb.DT_FLOAT, contents[i])
      i_offset = i_offset + 1
    for i in range(int64_count):
      pb_req.add_feed(input_names[i_offset], shapes[i_offset], tf_pb.DT_INT64, i64_contents[i])
      i_offset = i_offset + 1
    for i in range(double_count):
      pb_req.add_feed(input_names[i_offset], shapes[i_offset], tf_pb.DT_DOUBLE, d_contents[i])
      i_offset = i_offset + 1
    for i in range(int_count):
      pb_req.add_feed(input_names[i_offset], shapes[i_offset], tf_pb.DT_INT32, i_contents[i])
      i_offset = i_offset + 1
    for i in range(str_count):
      pb_req.add_feed(input_names[i_offset], shapes[i_offset], tf_pb.DT_STRING, s_contents[i])
      i_offset = i_offset + 1
    pb_buf = pb_req.to_string();

  pb_etime = time.time()

  pb_t = (pb_etime - pb_stime) / TESTING_COUNT * 1000
  print('pb encode time is {} ms'.format(pb_t), ", size = ", len(pb_buf))


  if DEBUG:
    # decode pb
    pb_recv_req = tf_pb.PredictRequest()
    pb_recv_req.ParseFromString(pb_buf)
    print(pb_recv_req.signature_name)
    recv_inputs = pb_recv_req.inputs
    pcount = 0
    for k, v in recv_inputs.items():
      pcount = pcount + 1
      print (k, "  ", v)
    print (pcount)
    for name in pb_recv_req.output_filter:
      print (name)

    # write buf to file, will be check by cpp programe
    #
    with open("/tmp/request_buf.pb", mode="wb") as f:
      f.write(pb_buf)

  print (str_rand(100))


if __name__ == '__main__':
  main()

