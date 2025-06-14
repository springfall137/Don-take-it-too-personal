import { connectDB } from "@/util/database.js"
import { ObjectId } from "mongodb"
import Comment from "./Comment"

export default async function Detail(props) {
    const { id } = await props.params;
    const db = (await connectDB).db("forum");
    const result = await db.collection('post').findOne({ _id: new ObjectId(id) });
    // console.log(props)
    return (
        <div>
            <h4>상세페이지</h4>
            <h4>{result.title }</h4>
            <p>{result.content}</p>
            <Comment _id={result._id.toString()} />
        </div>
    )
}
