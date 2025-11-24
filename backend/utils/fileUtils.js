// utils/fileUtils.js
import fs from "fs";

export const deleteFile = (filePath) => {
  try {
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
  } catch (err) {
    console.error("⚠️ Error deleting file:", err.message);
  }
};
