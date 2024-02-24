const mongoose = require("mongoose");

require("dotenv").config();

const connectDB = async () => {
  //console.log(process.env.MONGO);
  try {
    const conn = await mongoose.connect(process.env.MONGO);

    console.log(`MongoDB Connected: ${conn.connection.host}`);
  } catch (error) {
    console.log(`Error is ${error.message}`);
    process.exit(1);
  }
};

module.exports = connectDB;
