def test_preview(self):
    signal = abstracts.Signal1D.from_csv('tests/data/table_2.csv')
    # 断言以下代码运行时会出现 TypeError
    with self.assertRaises(TypeError):
        signal.preview(export_path="tests/out/table_2_preview_with_Signal1D") # No curve will be observed because the plot_at method is not implemented in the abstract class