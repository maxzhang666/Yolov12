from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
workspace = rf.workspace("YOUR_WORKSPACE")

workspace.deploy_model(
  model_type="yolov8",
  model_path="./runs/train/weights",
  project_ids=["project-1", "project-2", "project-3"],
  model_name="my-custom-model"
)