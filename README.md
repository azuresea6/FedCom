# FedCom
 a robust, safe and adaptive mechanism for adjusting communication frequency
 # Enhancing Data Privacy Protection and Efficiency in Distributed Learning Through Adaptive Communication Techniques

**Author:** Xiong Zhenxiang  
**Affiliation:** Michigan Institute, Shanghai Jiao Tong University, Shanghai, 200240, China  
**Email:** azuresea@sjtu.edu.cn

## Abstract
With the gradual proliferation of smart devices in recent years, distributed learning has become increasingly common and important in various practical scenarios. However, the time consumption during the communication process has also become a significant issue that cannot be overlooked. This paper collects runtime data from different devices and utilizes a biased label predictor based on recursive simulation principles to create a dataset for pre-training various adaptive communication algorithms that dynamically adjust the number of local training rounds. The performance of this adaptive communication algorithm is tested in optimizing the efficiency of a local update SGD algorithm simulated to run on four different devices, as well as its adaptive effect on the local update rounds parameter. To enhance the robustness of our algorithm, we also introduced an algorithm update mechanism and compared it with our pre-trained model. 

## Keywords
- Distributed Learning
- Adaptive Communication Algorithms
- Local Training Rounds
- Strong Adaptability
- Privacy Security

## Introduction
The importance of distributed learning is increasingly evident in today's world, especially in sectors like healthcare and finance, where federated learning effectively protects sensitive data while enabling model training. However, challenges such as user system heterogeneity and varying network conditions affect the efficiency of distributed training. This study aims to introduce a secure and robust communication algorithm for federated learning that can dynamically adjust the number of local training rounds, providing a universal and effective communication mechanism for practical deployment.

## Methodology

### 1. Data Preparation
This study utilizes four different devices (two 3070 Ti GPUs, one 3070, and one 3060) to simulate performance differences in real deployment scenarios. The data is generated through a four-threaded local update SGD function, recording average computation times for each communication cycle. Data quality is enhanced through preprocessing, including filling NA values and generating labels based on recursive simulation principles.

### 2. Data Generation
- **`data_cre` file**: Used to generate the dataset. It needs to be run on four different devices, and the resulting `.xlsx` files should be consolidated.
- **`data_dealer` file**: Contains our preliminary data processing methods and label generators. Run the relevant functions as needed to produce the final dataset.

### 3. Communication and Main Algorithm
- **`basic_method` file**: Defines the basic methods for communication algorithms.
- **`ULSGD` and `ULSGD_UP` files**: Define the pre-trained communication algorithm and the algorithm that updates during training. We encapsulate these in the `test` file, and during runtime, you only need to modify the calls in the `test` file and the index of the method you want to use after generating the final dataset.
- **`evaluate` file**: Defines all baseline methods for evaluating the performance of different algorithms.
- **`plot` file**: Used for data visualization of the generated reports.

## Results
The experimental results indicate that different communication algorithms yield varying levels of accuracy and computational efficiency. The self-updating communication algorithm shows optimization compared to pre-trained models, although it increases communication time. The study also highlights the trade-offs between generalization capability and operational efficiency.

## Discussion
The analysis reveals that while dynamic communication models can improve accuracy and computational efficiency, they also introduce time costs that may reduce overall efficiency. Recommendations include using pre-trained models in uniform environments and adopting dynamic updating mechanisms in scenarios requiring higher generalization capabilities.

## Conclusion
This research underscores the importance of adaptive communication algorithms in distributed learning. By strategically selecting appropriate models based on specific requirements, optimal performance can be achieved while maintaining computational efficiency.

## How to Run the Project

1. **Data Generation**:
   - Run the `data_cre` file on four different devices to generate the dataset. Collect the resulting `.xlsx` files and consolidate them for further processing.

2. **Data Processing**:
   - Use the functions in the `data_dealer` file to preprocess the dataset and generate labels as needed. This step is crucial for preparing the final dataset.

3. **Communication Algorithms**:
   - Implement the basic communication methods found in the `basic_method` file.
   - Utilize the `ULSGD` and `ULSGD_UP` files for the pre-trained communication algorithm and the adaptive algorithm that updates during training. 

4. **Testing**:
   - Modify the `test` file to call the appropriate functions and set the desired method index after generating the final dataset.

5. **Evaluation**:
   - Use the `evaluate` file to run all baseline methods and assess the performance of your algorithms.

6. **Visualization**:
   - Generate visual reports using the `plot` file to analyze the results and present findings effectively.

## References
1. Sheller, M. J., et al. "Federated Learning in Medicine: A Systematic Review." arXiv preprint arXiv:2004.00445, 2020.
2. Kairouz, P., et al. "Advances and Open Problems in Federated Learning." arXiv preprint arXiv:1912.04977, 2021.
3. Li, T., et al. "Federated Learning: Challenges, Methods, and Future Directions." IEEE Signal Processing Magazine, 2020.
4. Mao, Y., and J. Zhang. "Federated Learning with Heterogeneous Data: A Survey." IEEE Transactions on Neural Networks and Learning Systems, 2022.
5. Liu, Y., and S. Wang. "Communication-Efficient Federated Learning: A Survey." IEEE Transactions on Neural Networks and Learning Systems, 2022.
6. Luo, L., et al. "Communication-Efficient Federated Learning With Adaptive Aggregation." IEEE Transactions on Services Computing, 2024.
7. Yang, T., et al. "Byzantine-Resilient Federated Learning through Adaptive Model Aggregation." arXiv preprint arXiv:1902.01046, 2019.
8. Vaidya, N. H. "Security and Privacy for Distributed Optimization & Distributed Machine Learning." PODC'21, 2021.
9. Zhou, Z., & Pei, J. "Privacy-Preserving Distributed Data Mining: A Survey." IEEE Transactions on Knowledge and Data Engineering, 2010.
10. McMahan, B. F., et al. "Communication-efficient learning of deep networks from decentralized data." AISTATS, 2017.
11. Lin, T., et al. "Don't Use Large Mini-Batches, Use Local SGD." arXiv Preprint arXiv:1808.07217, 2018.
