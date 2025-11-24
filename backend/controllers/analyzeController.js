// controllers/analyzeController.js
import axios from "axios";
import fs from "fs";
import FormData from "form-data";
import { PYTHON_SERVICE_URL } from "../config/pythonService.js";
import { deleteFile } from "../utils/fileUtils.js";

export const analyzeExercise = async (req, res) => {
  try {
    const { exercise_name } = req.body;
    const videoPath = req.file?.path;

    if (!videoPath) {
      return res.status(400).json({ error: "Video file not found in request" });
    }

    console.log(`📤 Sending '${req.file.filename}' for ${exercise_name} analysis...`);

    // Prepare form data for FastAPI
    const formData = new FormData();
    formData.append("exercise_name", exercise_name);
    formData.append("video", fs.createReadStream(videoPath));

    // Send to FastAPI
    const response = await axios.post(PYTHON_SERVICE_URL, formData, {
      headers: formData.getHeaders(),
      maxBodyLength: Infinity,
    });

    // Delete uploaded file after sending
    deleteFile(videoPath);

    console.log("✅ Response received from Python service");

    // --- Utility: convert to clean relative path
    const makeRelative = (p) => {
      if (!p) return null;
      const strPath =
        typeof p === "string"
          ? p
          : typeof p === "object" && p.path
          ? p.path
          : null;
      if (!strPath) return null;
      return "/" + strPath.replace(/^[\\/]+/, "").replace(/\\/g, "/");
    };

    // Extract data
    const data = response.data || {};
    const files = data.files || {};

    const videoFiles = files.videos || {};
    const plotFiles = files.plots || {};
    const csvPath = files.csv;

    // Optional feedback metadata from Python (if present)
    const ai_form_score = data.ai_form_score ?? 0;
    const total_reps = data.total_reps ?? 0;
    const feedback = data.feedback ?? [];

    // ✅ Return only relative paths
    res.status(200).json({
      message: "Analysis complete",
      ai_form_score,
      total_reps,
      feedback,
      files: {
        videos: {
          annotated_full: makeRelative(videoFiles.annotated_full),
          annotated_simple: makeRelative(videoFiles.annotated_simple),
          stick_figure: makeRelative(videoFiles.stick_figure),
        },
        plots: {
          angle_progression: makeRelative(plotFiles.angle_progression),
          error_summary: makeRelative(plotFiles.error_summary),
          quality_plot: makeRelative(plotFiles.quality_plot),
        },
        csv: makeRelative(csvPath),
      },
    });
  } catch (err) {
    console.error("❌ Error in analyzeExercise:", err.message);
    if (err.response?.data) console.error("Python response error:", err.response.data);
    res.status(500).json({
      error: "Internal server error",
      details: err.message,
    });
  }
};
