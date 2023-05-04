from tester import ModelTester

if __name__ == "__main__":
    
    for recipe in ['embAug', 'eda', 'wordnet', 'charswap']:
        print(f"================ {recipe}  =================")
        tester = ModelTester(
            test_file = f"../dataAugmentation/output/test_{recipe}_1000_1.json",
            test_model_path = "models/best_model_v1.pt",
        )
        tester.do_evaluate()
