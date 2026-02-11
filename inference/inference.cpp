#include <torch/script.h>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

std::vector<std::string> load_class_names(const std::string& path) {
    std::ifstream file(path);
    nlohmann::json j;
    file >> j;
    return j["label_encoder"].get<std::vector<std::string>>();
}

int main() {
    auto class_names = load_class_names("/home/azureuser/simulator/outputs/models/active_model_metadata.json");
    
    torch::jit::script::Module module = torch::jit::load("/home/azureuser/simulator/outputs/models/active_model_optimized.pth");
    module.eval();
    
    std::vector<float> input_data(128, 0.5f);
    
    torch::Tensor input = torch::from_blob(input_data.data(), {1, 128});
    auto logits = module.forward({input}).toTensor();
    auto probs = torch::softmax(logits, 1);
    
    auto max_result = probs.max(1);
    int predicted_idx = std::get<1>(max_result).item<int>();
    float probability = std::get<0>(max_result).item<float>();
    
    std::cout << "Predicted: " << class_names[predicted_idx] 
              << " (" << probability * 100 << "%)" << std::endl;
    
    return 0;
}