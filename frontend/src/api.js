import axios from "axios";

const API_BASE = "http://localhost:5000/api"; // Node backend endpoint

export const uploadExerciseVideo = async (exerciseName, file) => {
  const formData = new FormData();
  formData.append("exercise_name", exerciseName);
  formData.append("video", file);

  const response = await axios.post(`${API_BASE}/analyze`, formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
};
