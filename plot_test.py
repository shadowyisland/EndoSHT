from depthnet.meters import plot_metrics
import json

json_path = r"e:\CodeWork\PyWork\MonoLoT\results\<exp_name>\metrics.json"
out_pdf   = r"e:\CodeWork\PyWork\MonoLoT\results\<exp_name>\metrics.pdf"

with open(json_path, "r") as f:
    data = json.load(f)

plot_metrics(data, pdf_path=out_pdf)