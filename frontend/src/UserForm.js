import axios from "axios";
import React, { useState } from "react";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import LoadingSpinner from "./LoadingSpinner";

const client = axios.create({
  baseURL: "http://localhost:5000",
});

export default function UserForm() {
  const [predictionResult, setPredictionResult] = React.useState(null);
  const [errorMsg, setErrorMsg] = React.useState(null);
  const [param1, setParam1] = useState(null);
  const [param2, setParam2] = useState(null);
  const [param3, setParam3] = useState(null);
  const [param4, setParam4] = useState(null);
  const [param5, setParam5] = useState(null);
  const [param6, setParam6] = useState(null);
  const [param7, setParam7] = useState(null);
  const [param8, setParam8] = useState(null);
  const [trainAccuracy, setTrainAccuracy] = useState(null);
  const [testAccuracy, setTestAccuracy] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const HandleSubmit = (event) => {
    console.log("HandleSubmit ran");
    setPredictionResult("");
    setTrainAccuracy("");
    setTestAccuracy("");
    setIsLoading(true);

    event.preventDefault();

    setParam1(event.target.jumlahLengan.value);
    setParam2(event.target.cabangLengan.value);
    setParam3(event.target.bentukMorfologi.value);
    setParam4(event.target.knob.value);
    setParam5(event.target.ukuranLengan.value);
    setParam6(event.target.bentukLengan.value);
    setParam7(event.target.bentukUjungLengan.value);
    setParam8(event.target.bentukUjungLenganMelengkung.value);

    // access input values using name prop
    console.log("Jumlah Lengan:", event.target.jumlahLengan.value);
    console.log("Cabang Lengan:", event.target.cabangLengan.value);
    console.log("Bentuk Morfologi:", event.target.bentukMorfologi.value);
    console.log("Knob:", event.target.knob.value);
    console.log("Ukuran Lengan:", event.target.ukuranLengan.value);
    console.log("Bentuk Lengan:", event.target.bentukLengan.value);
    console.log("Bentuk Ujung Lengan:", event.target.bentukUjungLengan.value);
    console.log(
      "Bentuk Ujung Lengan Melengkung:",
      event.target.bentukUjungLenganMelengkung.value
    );

    // clear all input values in the form
    event.target.reset();

    client
      .get(
        `/api/prediction?jumlah_lengan=${param1}&cabang_lengan=${param2}&bentuk_morfologi=${param3}&knob=${param4}&ukuran_lengan=${param5}&bentuk_lengan=${param6}&bentuk_ujung_lengan=${param7}&bentuk_ujung_lengan_melengkung=${param8}`
      )
      .then((response) => {
        setPredictionResult(response.data.prediction_class);
        setTrainAccuracy(response.data.train_accuracy.concat("%"));
        setTestAccuracy(response.data.test_accuracy.concat("%"));
        setIsLoading(false);
      })
      .catch((error) => {
        setErrorMsg(error);
        console.log("Error:", errorMsg);
        setIsLoading(false);
      });
  };

  const result = (
    <Row>
      <p>Prediction : {predictionResult}</p>
      <p>Train Accuracy : {trainAccuracy}</p>
      <p>Test Accuracy : {testAccuracy}</p>
    </Row>
  );

  return (
    <Container className="form">
      <Row>
        <br />
      </Row>
      <Row>
        <br />
      </Row>
      <p>Nannofossil Classifier</p>
      <Row>
        <Form onSubmit={HandleSubmit}>
          <Row>
            <Form.Group
              onChange={(e) => {
                setParam1(e.target.value);
              }}
            >
              <Row>
                <Col>
                  <p>Jumlah Lengan</p>
                </Col>
                <Col>
                  <Form.Check
                    inline
                    label="3"
                    name="jumlahLengan"
                    type="radio"
                    id={"jumlah-lengan-3"}
                    value="3"
                  />
                  <Form.Check
                    inline
                    label="4"
                    name="jumlahLengan"
                    type="radio"
                    id={"jumlah-lengan-4"}
                    value="4"
                  />
                  <Form.Check
                    inline
                    label="5"
                    name="jumlahLengan"
                    type="radio"
                    id={"jumlah-lengan-5"}
                    value="5"
                  />
                  <Form.Check
                    inline
                    label="6"
                    name="jumlahLengan"
                    type="radio"
                    id={"jumlah-lengan-6"}
                    value="6"
                  />
                </Col>
              </Row>
            </Form.Group>
          </Row>
          <Row>
            <Form.Group
              onChange={(e) => {
                setParam2(e.target.value);
              }}
            >
              <Row>
                <Col>
                  <p>Jumlah Lengan</p>
                </Col>
                <Col>
                  <Form.Check
                    inline
                    label="Ada"
                    name="cabangLengan"
                    type="radio"
                    id={"cabang-lengan-1"}
                    value="1"
                  />
                  <Form.Check
                    inline
                    label="Tidak"
                    name="cabangLengan"
                    type="radio"
                    id={"cabang-lengan-0"}
                    value="0"
                  />
                </Col>
              </Row>
            </Form.Group>
          </Row>
          <Row>
            <Form.Group
              onChange={(e) => {
                setParam3(e.target.value);
              }}
            >
              <Row>
                <Col>
                  <p>Bentuk Morfologi</p>
                </Col>
                <Col>
                  <Form.Check
                    inline
                    label="Simetris"
                    name="bentukMorfologi"
                    type="radio"
                    id={"bentuk-morfologi-1"}
                    value="1"
                  />
                  <Form.Check
                    inline
                    label="Asimetris"
                    name="bentukMorfologi"
                    type="radio"
                    id={"bentuk-morfologi-0"}
                    value="0"
                  />
                </Col>
              </Row>
            </Form.Group>
          </Row>
          <Row>
            <Form.Group
              onChange={(e) => {
                setParam4(e.target.value);
              }}
            >
              <Row>
                <Col>
                  <p>Knob</p>
                </Col>
                <Col>
                  <Form.Check
                    inline
                    label="Ada"
                    name="knob"
                    type="radio"
                    id={"knob-1"}
                    value="1"
                  />
                  <Form.Check
                    inline
                    label="Tidak"
                    name="knob"
                    type="radio"
                    id={"knob-0"}
                    value="0"
                  />
                </Col>
              </Row>
            </Form.Group>
          </Row>
          <Row>
            <Form.Group
              onChange={(e) => {
                setParam5(e.target.value);
              }}
            >
              <Row>
                <Col>
                  <p>Ukuran Lengan</p>
                </Col>
                <Col>
                  <Form.Check
                    inline
                    label="Panjang"
                    name="ukuranLengan"
                    type="radio"
                    id={"ukuran-lengan-1"}
                    value="1"
                  />
                  <Form.Check
                    inline
                    label="Pendek"
                    name="ukuranLengan"
                    type="radio"
                    id={"ukuran-lengan-0"}
                    value="0"
                  />
                </Col>
              </Row>
            </Form.Group>
          </Row>
          <Row>
            <Form.Group
              onChange={(e) => {
                setParam6(e.target.value);
              }}
            >
              <Row>
                <Col>
                  <p>Bentuk Lengan</p>
                </Col>
                <Col>
                  <Form.Check
                    inline
                    label="Pipih"
                    name="bentukLengan"
                    type="radio"
                    id={"bentuk-lengan-0"}
                    value="0"
                  />
                  <Form.Check
                    inline
                    label="Tebal"
                    name="bentukLengan"
                    type="radio"
                    id={"bentuk-lengan-1"}
                    value="1"
                  />
                </Col>
              </Row>
            </Form.Group>
          </Row>
          <Row>
            <Form.Group
              onChange={(e) => {
                setParam7(e.target.value);
              }}
            >
              <Row>
                <Col>
                  <p>Bentuk Ujung Lengan</p>
                </Col>
                <Col>
                  <Form.Check
                    inline
                    label="Lancip"
                    name="bentukUjungLengan"
                    type="radio"
                    id={"bentuk-ujung-lengan-1"}
                    value="1"
                  />
                  <Form.Check
                    inline
                    label="Tumpul"
                    name="bentukUjungLengan"
                    type="radio"
                    id={"bentuk-ujung-lengan-0"}
                    value="0"
                  />
                </Col>
              </Row>
            </Form.Group>
          </Row>
          <Row>
            <Form.Group
              onChange={(e) => {
                setParam8(e.target.value);
              }}
            >
              <Row>
                <Col>
                  <p>Bentuk Ujung Lengan Melengkung</p>
                </Col>
                <Col>
                  <Form.Check
                    inline
                    label="Ya"
                    name="bentukUjungLenganMelengkung"
                    type="radio"
                    id={"bentuk-ujung-lengan-melengkung-1"}
                    value="1"
                  />
                  <Form.Check
                    inline
                    label="Tidak"
                    name="bentukUjungLenganMelengkung"
                    type="radio"
                    id={"bentuk-ujung-lengan-melengkung-0"}
                    value="0"
                  />
                </Col>
              </Row>
            </Form.Group>
          </Row>
          <Button variant="custom" type="submit" disabled={isLoading}>
            Predict Class
          </Button>
        </Form>
      </Row>
      <Row>
        <p></p>
      </Row>
      {isLoading ? <LoadingSpinner /> : result}
    </Container>
  );
}
