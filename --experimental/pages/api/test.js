export default function handler(request, response) {
    if (request.method == 'POST') {
        console.log(123)
        return response.status(200).json('처리완료')
    }

}
