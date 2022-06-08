from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
import os
#from werkzeug import secure_filename
app = Flask(__name__)

# picFolder = os.path.join('static', 'images')

# app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/success//<name>')
def success(name):
    # dimensions of our images
    img_width, img_height = 224, 224

    # load the model we saved
    model = load_model('keras.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # predicting images
    #img = image.load_img(name).convert('L')
    #img = img.resize(img_height, img_width)
    img = image.load_img(name, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)

    if classes[0][0] == 1:

        return '''
        <!DOCTYPE html>
      <html>
        <head>
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <style>

             body {
                  font-family: "Poppins", "Arial", "Helvetica Neue", sans-serif;
                  font-weight: 400;
                  font-size: 14px;
                  background: -webkit-linear-gradient(bottom, #fbc2eb 0%, #a18cd1 100%);
        background: -moz-linear-gradient(bottom, #fbc2eb 0%, #a18cd1 100%);
        background: -o-linear-gradient(bottom, #fbc2eb 0%, #a18cd1 100%);
        background: linear-gradient(to top, #fbc2eb 0%, #a18cd1 100%);
              }
            * {
              box-sizing: border-box;
            }

            label {
              padding: 12px 12px 12px 0;
              display: inline-block;
            }

            .container {
              border-radius: 5px;
              background-color: #f2f2f2;
              padding: 20px;
              margin-top: 5%;
              font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
              font-size: large;
            }

            .heading{
              border-radius: 5px;
              background-color: #f2f2f2;
              padding: 20px;
              margin-top: 10%;
              font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
              
            }
            .heading1{
              border-radius: 5px;
              background-color: #f2f2f2;
              padding: 20px;
              margin-top: 5%;
              font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
              
            }

                  #res{
             color: green; 
            }


            .col-25 {
              float: left;
              width: 25%;
              margin-top: 6px;
            }

            .col-75 {
              float: left;
              width: 75%;
              margin-top: 6px;
            }

            /* Clear floats after the columns */
            .row:after {
              content: "";
              display: table;
              clear: both;
            }

            /* Responsive layout - when the screen is less than 600px wide, make the two columns stack on top of each other instead of next to each other */
            @media screen and (max-width: 600px) {
              .col-25,
              .col-75,
              input[type="submit"] {
                width: 100%;
                margin-top: 0;
              }
            }
            #main-footer {
        text-align: center;
        padding: 1rem;
        background: darken($primary-color, 10);
        color: set-text-color($primary-color);
        height: 60px;
      }
          </style>
        </head>
        <body>
          
          <div class="heading">
            <h1>REPORT</h1>
          </div>
          <div class="container">
            <form action="/action_page.php">
              
              <div class="row">
                <div class="col-25">
                  <label for="country">Result</label>
                </div>
                <div class="col-75">
                  <label id="res" name="res">
                    NEGATIVE
                  </select>
                </div>
              </div>
              <div class="row">
                <div class="col-25">
                  <label for="subject">Disclaimer</label>
                </div>
                <div class="col-75">
                  <label
                    id="subject"
                    name="subject"
                    style="height: 200px;">
                    This is a report generated by a Convolutional Neural Network Model which was trained by real images provided by authorities of Corona Positive and Negative examples.
                    The accuracy of model is above 82%. But according to research there are some positive cases which do not show symptoms in chest X-Rays. Hence
                    its recomended to consult a doctor before any treatment.
                  </label>
                </div>
              </div>
              
            </form>
          </div>




        </body>
        <footer id="main-footer">tek-up 2021<br /></footer>
      </html>


               '''
    else:

        return '''
        <!DOCTYPE html>
<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>

               body {
            font-family: "Poppins", "Arial", "Helvetica Neue", sans-serif;
            font-weight: 400;
            font-size: 14px;
            background: -webkit-linear-gradient(bottom, #fbc2eb 0%, #a18cd1 100%);
  background: -moz-linear-gradient(bottom, #fbc2eb 0%, #a18cd1 100%);
  background: -o-linear-gradient(bottom, #fbc2eb 0%, #a18cd1 100%);
  background: linear-gradient(to top, #fbc2eb 0%, #a18cd1 100%);
        }
      * {
        box-sizing: border-box;
      }

      label {
        padding: 12px 12px 12px 0;
        display: inline-block;
      }

      .container {
        border-radius: 5px;
        background-color: #f2f2f2;
        padding: 20px;
        margin-top: 5%;
        font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
        font-size: large;
      }

      .heading{
        border-radius: 5px;
        background-color: #f2f2f2;
        padding: 20px;
        margin-top: 10%;
        font-family: 'Gill Sans', 'Gill Sans MT', Calibri, 'Trebuchet MS', sans-serif;
        
      }

                  #res{
       color: red; 
      }


      .col-25 {
        float: left;
        width: 25%;
        margin-top: 6px;
      }

      .col-75 {
        float: left;
        width: 75%;
        margin-top: 6px;
      }

      /* Clear floats after the columns */
      .row:after {
        content: "";
        display: table;
        clear: both;
      }

      /* Responsive layout - when the screen is less than 600px wide, make the two columns stack on top of each other instead of next to each other */
      @media screen and (max-width: 600px) {
        .col-25,
        .col-75,
        input[type="submit"] {
          width: 100%;
          margin-top: 0;
        }
      }
      #main-footer {
  text-align: center;
  padding: 1rem;
  background: darken($primary-color, 10);
  color: set-text-color($primary-color);
  height: 60px;
}
    </style>
  </head>
  <body>
    
    <div class="heading">
      <h1>REPORT</h1>
    </div>
    <div class="container">
      <form action="/action_page.php">
        
        <div class="row">
          <div class="col-25">
            <label for="country">Result</label>
          </div>
          <div class="col-75">
            <label id="res" name="res">
              POSITIVE
            </select>
          </div>
        </div>
        <div class="row">
          <div class="col-25">
            <label for="subject">Disclaimer</label>
          </div>
          <div class="col-75">
            <label
              id="subject"
              name="subject"
              style="height: 200px;">
              This is a report generated by a Convolutional Neural Network Model which was trained by real images provided by authorities of Corona Positive and Negative examples.
              The accuracy of model is above 82%. But according to research there are some positive cases which do not show symptoms in chest X-Rays. Hence
              its recomended to consult a doctor before any treatment.
            </label>
          </div>
        </div>


        
      </form>
    </div>

    

    
  </body>


  <footer id="main-footer">tek_up 2021<br /></footer>
</html>


               '''


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        file = request.files['nm']
        basepath = os.path.dirname(__file__)
        #file.save(os.path.join(basepath, "uploads", file.filename))
        #user = os.path.join(basepath, "uploads", file.filename)
        file.save(os.path.join(basepath, file.filename))
        user = file.filename
        return redirect(url_for('success', name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))


@app.route("/") 
def home_view():
        return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)