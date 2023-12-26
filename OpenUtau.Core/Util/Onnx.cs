using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;
using OpenUtau.Core.Util;
using Vortice.DXGI;

namespace OpenUtau.Core {
    public class GpuInfo {
        public int deviceId;
        public string description = "";

        override public string ToString() {
            return $"[{deviceId}] {description}";
        }
    }
    public abstract class IOnnxInferenceSession : IDisposable {
        public InferenceSession session;
        public abstract IReadOnlyCollection<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs);
        public abstract void Dispose();
    }

    class LocalInferenceSession : IOnnxInferenceSession {
        private bool _disposed = false;
        public LocalInferenceSession(InferenceSession session) {
            this.session = session;
        }

        public override IReadOnlyCollection<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs) {
            return session.Run(inputs);
        }

        ~LocalInferenceSession() {
            Dispose(false);
        }
        public override void Dispose() {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        protected virtual void Dispose(bool disposing) {
            if (!_disposed) {
                if (disposing) {
                    session.Dispose();
                }
                _disposed = true;
            }
        }
    }

    class RemoteInferenceSession : IOnnxInferenceSession {
        class RemoteInput {
            public string type; // Not used for now
            public int[] shape;
            public byte[]? uint8_data;
            public UInt16[]? uint16_data;
            public UInt32[]? uint32_data;
            public UInt64[]? uint64_data;
            public sbyte[]? int8_data;
            public Int16[]? int16_data;
            public Int32[]? int32_data;
            public Int64[]? int64_data;
            public float[]? float_data;
            public Double[]? double_data;
            public Float16[]? float16_data;
            public BFloat16[]? bfloat16_data;
            public bool[]? bool_data;
        }

        private bool _disposed = false;
        string url;
        string modelPath;
        public RemoteInferenceSession(string url, string modelPath, InferenceSession session) {
            this.url = url;
            this.modelPath = modelPath;
            this.session = session;
        }

        public override IReadOnlyCollection<NamedOnnxValue> Run(IReadOnlyCollection<NamedOnnxValue> inputs) {
            // Convert inputs to RemoteInput
            Dictionary<string, RemoteInput> remoteInputs = new Dictionary<string, RemoteInput>();

            foreach (NamedOnnxValue input in inputs) {
                Type typeT = input.Value.GetType().GetGenericArguments()[0];
                if (typeT == typeof(byte)) {
                    var tensor = input.AsTensor<byte>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(uint8)",
                        shape = tensor.Dimensions.ToArray(),
                        uint8_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(UInt16)) {
                    var tensor = input.AsTensor<ushort>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(uint16)",
                        shape = tensor.Dimensions.ToArray(),
                        uint16_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(UInt32)) {
                    var tensor = input.AsTensor<uint>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(uint32)",
                        shape = tensor.Dimensions.ToArray(),
                        uint32_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(UInt64)) {
                    var tensor = input.AsTensor<ulong>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(uint64)",
                        shape = tensor.Dimensions.ToArray(),
                        uint64_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(sbyte)) {
                    var tensor = input.AsTensor<sbyte>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(int8)",
                        shape = tensor.Dimensions.ToArray(),
                        int8_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(Int16)) {
                    var tensor = input.AsTensor<short>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(int16)",
                        shape = tensor.Dimensions.ToArray(),
                        int16_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(Int32)) {
                    var tensor = input.AsTensor<int>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(int32)",
                        shape = tensor.Dimensions.ToArray(),
                        int32_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(Int64)) {
                    var tensor = input.AsTensor<long>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(int64)",
                        shape = tensor.Dimensions.ToArray(),
                        int64_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(float)) {
                    var tensor = input.AsTensor<float>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(float)",
                        shape = tensor.Dimensions.ToArray(),
                        float_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(Double)) {
                    var tensor = input.AsTensor<double>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(double)",
                        shape = tensor.Dimensions.ToArray(),
                        double_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(Float16)) {
                    var tensor = input.AsTensor<Float16>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(float16)",
                        shape = tensor.Dimensions.ToArray(),
                        float16_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(BFloat16)) {
                    var tensor = input.AsTensor<BFloat16>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(bfloat16)",
                        shape = tensor.Dimensions.ToArray(),
                        bfloat16_data = tensor.ToArray()
                    });
                } else if (typeT == typeof(bool)) {
                    var tensor = input.AsTensor<bool>();
                    remoteInputs.Add(input.Name, new RemoteInput {
                        type = "tensor(bool)",
                        shape = tensor.Dimensions.ToArray(),
                        bool_data = tensor.ToArray()
                    });
                } else {
                    throw new Exception($"Unsupported type {typeT}");
                }
            }

            // Send request
            string reqStr = JsonConvert.SerializeObject(remoteInputs);
            string apiUrl = url + "/inference/" + HttpUtility.UrlEncode(modelPath);

            using (var client = new System.Net.WebClient()) {
                client.Headers.Add("Content-Type", "application/json");
                client.Encoding = System.Text.Encoding.UTF8;
                string resStr = client.UploadString(apiUrl, reqStr);
                var resData = JsonConvert.DeserializeObject<Dictionary<string, RemoteInput>>(resStr);
                var results = new List<NamedOnnxValue>(resData.Count);
                for (int i = 0; i < resData.Count; i++) {
                    var output = resData.ElementAt(i);
                    NamedOnnxValue onnxValue = null;
                    if (output.Value.type == "tensor(uint8)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<byte>(output.Value.uint8_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(uint16)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<ushort>(output.Value.uint16_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(uint32)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<uint>(output.Value.uint32_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(uint64)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<ulong>(output.Value.uint64_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(int8)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<sbyte>(output.Value.int8_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(int16)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<short>(output.Value.int16_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(int32)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<int>(output.Value.int32_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(int64)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<long>(output.Value.int64_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(float)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<float>(output.Value.float_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(double)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<double>(output.Value.double_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(float16)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<Float16>(output.Value.float16_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(bfloat16)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<BFloat16>(output.Value.bfloat16_data, output.Value.shape));
                    } else if (output.Value.type == "tensor(bool)") {
                        onnxValue = NamedOnnxValue.CreateFromTensor(output.Key, new DenseTensor<bool>(output.Value.bool_data, output.Value.shape));
                    } else {
                        throw new Exception("Unknown tensor type: " + output.Value.type);
                    }
                    results.Add(onnxValue);
                }
                return results;
            }
        }

        ~RemoteInferenceSession() {
            Dispose(false);
        }
        public override void Dispose() {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
        protected virtual void Dispose(bool disposing) {
            if (!_disposed) {
                _disposed = true;
            }
        }
    }

    public class Onnx {
        public static List<string> getRunnerOptions() {
            if (OS.IsWindows()) {
                return new List<string> {
                "cpu",
                "directml",
                "remote"
                };
            } else if (OS.IsMacOS()) {
                return new List<string> {
                "cpu",
                "coreml",
                "remote"
                };
            }
            return new List<string> {
                "cpu",
                "remote"
            };
        }

        public static List<GpuInfo> getGpuInfo() {
            List<GpuInfo> gpuList = new List<GpuInfo>();
            if (OS.IsWindows()) {
                DXGI.CreateDXGIFactory1(out IDXGIFactory1 factory);
                for(int deviceId = 0; deviceId < 32; deviceId++) {
                    factory.EnumAdapters1(deviceId, out IDXGIAdapter1 adapterOut);
                    if(adapterOut is null) {
                        break;
                    }
                    gpuList.Add(new GpuInfo {
                        deviceId = deviceId,
                        description = adapterOut.Description.Description
                    }) ;
                }
            }
            if (gpuList.Count == 0) {
                gpuList.Add(new GpuInfo {
                    deviceId = 0,
                });
            }
            return gpuList;
        }

        private static Tuple<string, SessionOptions> getOnnxSessionOptions(){
            SessionOptions options = new SessionOptions();
            List<string> runnerOptions = getRunnerOptions();
            string runner = Preferences.Default.OnnxRunner;
            if (String.IsNullOrEmpty(runner)) {
                runner = runnerOptions[0];
            }
            if (!runnerOptions.Contains(runner)) {
                runner = "cpu";
            }
            switch(runner){
                case "directml":
                    options.AppendExecutionProvider_DML(Preferences.Default.OnnxGpu);
                    break;
                case "coreml":
                    options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                    break;
            }
            return new Tuple<string, SessionOptions>(runner, options);
        }

        public static InferenceSession getInferenceSession(byte[] model) {
            var (_, options) = getOnnxSessionOptions();
            return new InferenceSession(model, options);
        }

        public static IOnnxInferenceSession getInferenceSession(string modelPath) {
            var (runner, options) = getOnnxSessionOptions();
            if (runner == "remote") {
                return new RemoteInferenceSession(Preferences.Default.OnnxRemoteUrl, modelPath, new InferenceSession(modelPath, options));
            } else {
                return new LocalInferenceSession(new InferenceSession(modelPath, options));
            }
        }

        public static void VerifyInputNames(InferenceSession session, IEnumerable<NamedOnnxValue> inputs) {
            var sessionInputNames = session.InputNames.ToHashSet();
            var givenInputNames = inputs.Select(v => v.Name).ToHashSet();
            var missing = sessionInputNames
                .Except(givenInputNames)
                .OrderBy(s => s, StringComparer.InvariantCulture)
                .ToArray();
            if (missing.Length > 0) {
                throw new ArgumentException("Missing input(s) for the inference session: " + string.Join(", ", missing));
            }
            var unexpected = givenInputNames
                .Except(sessionInputNames)
                .OrderBy(s => s, StringComparer.InvariantCulture)
                .ToArray();
            if (unexpected.Length > 0) {
                throw new ArgumentException("Unexpected input(s) for the inference session: " + string.Join(", ", unexpected));
            }
        }
    }
}
