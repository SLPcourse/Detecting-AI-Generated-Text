from tester import ModelTester

if __name__ == "__main__":
    
    for model_v in ['textfooler', 'textbugger', 'pwws', 'deepwordbug']:
        for recipe in ['textfooler', 'textbugger', 'pwws', 'deepwordbug']:
        print(f"================ {recipe}  =================")
        tester = ModelTester(
            test_file = f"../attack/outputs/test/{recipe}_1000_1.json",
            test_model_path = "models/best_model_v1.pt",
        )
        tester.do_evaluate()
