# ribs_suppression_prod
A repo for web-interface of handling bone suppression model.

1. Install requirements:  
`pip install -r requirements.txt`

2. Download a model () and put it into a folder `repo_folder/models/fpn_last_model.pth.tar`

3. Run the web-interface by `PYTHONPATH=. python src/backend/api.py`

4. In your web-browser go to the localhost:5005 and upload image file you want to process (.jpg, .jpeg and .png allowed)

5. Click "Submit" button and get the result!

6. To choose another image go back to previous page.

For better performance one should have NVIDIA GPU and CUDA installed on the machine.
