________/\\\\\\\\\____________/\\\_____/\\\\\\\\\\\\_______/\\\\\\\\\_________/\\\\\\\\\\\\_____/\\\\\\\\\\\______________________/\\\\\\_______________________________________        
 _____/\\\////////___________/\\\\\____\/\\\////////\\\___/\\\///////\\\_____/\\\//////////____/\\\/////////\\\___________________\////\\\_______________________________________       
  ___/\\\/__________________/\\\/\\\____\/\\\______\//\\\_\///______\//\\\___/\\\______________\//\\\______\///_______________________\/\\\_____/\\\_____/\\\_____________________      
   __/\\\__________________/\\\/\/\\\____\/\\\_______\/\\\___________/\\\/___\/\\\____/\\\\\\\___\////\\\______________________________\/\\\____\///___/\\\\\\\\\\\_____/\\\\\\\\__     
    _\/\\\________________/\\\/__\/\\\____\/\\\_______\/\\\________/\\\//_____\/\\\___\/////\\\______\////\\\___________________________\/\\\_____/\\\_\////\\\////____/\\\/////\\\_    
     _\//\\\_____________/\\\\\\\\\\\\\\\\_\/\\\_______\/\\\_____/\\\//________\/\\\_______\/\\\_________\////\\\________________________\/\\\____\/\\\____\/\\\_______/\\\\\\\\\\\__   
      __\///\\\__________\///////////\\\//__\/\\\_______/\\\____/\\\/___________\/\\\_______\/\\\__/\\\______\//\\\_______________________\/\\\____\/\\\____\/\\\_/\\__\//\\///////___  
       ____\////\\\\\\\\\___________\/\\\____\/\\\\\\\\\\\\/____/\\\\\\\\\\\\\\\_\//\\\\\\\\\\\\/__\///\\\\\\\\\\\/____/\\\\\\\\\\\\\\\__/\\\\\\\\\_\/\\\____\//\\\\\____\//\\\\\\\\\\_ 
        _______\/////////____________\///_____\////////////_____\///////////////___\////////////______\///////////_____\///////////////__\/////////__\///______\/////______\//////////__


# c4d2gs_lite Script Workflow

This repository includes a fast Cinema 4D script workflow to generate synthetic COLMAP data for Gaussian Splatting pipelines.

## What It Exports

Running the script produces:

- Synthetic COLMAP data files: `cameras.txt`, `images.txt`, `points3D.txt`
- Rendered image sequence from the animated render camera
- Optional `camera_poses.json` for NeRF-style tooling

## Quick Usage

1. Open Cinema 4D and select your target object.
2. Open [c4d2gs_lite.py](c4d2gs_lite.py) in Cinema 4D Script Manager.
3. Set your output path and basic capture settings near the top of the script.
4. Run the script to build the rig and export synthetic COLMAP data. Make sure your target object is selected in the object-manager; the generated cameras are always aimed at the selected object’s center axis.
5. Render the animation to output your frame sequence.
6. Import the synthetic COLMAP data and rendered images into your reconstruction/training app.

## Output Layout

Typical output folder structure:

- `cameras.txt`
- `images.txt`
- `points3D.txt`
- `camera_poses.json` (optional)
- `images/gs_0000.png`, `images/gs_0001.png`, ...

## Note About the Full Plugin

If you want a cleaner UI, richer controls, and a more guided production workflow, use the full C4D2GS plugin in [C4D2GS](http://Vaghabund.gumroad.com/l/c4d2gs). It is the easiest way to generate synthetic COLMAP data at scale with fewer manual steps.
